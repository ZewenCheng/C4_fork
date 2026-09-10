"""有界伙伴工具事务；先持久化尝试，恢复不能重复获得额度。"""
from pathlib import Path
import time
from report_contract import ContractError, atomic_json, digest_json, read_json, run_lock
from partner_protocol import VERSION


class PartnerCoordinator:
    def __init__(self, directory, context, tools, *, max_attempts=4, max_rounds=2):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 4 or type(max_rounds) is not int or not 1 <= max_rounds <= 2:
            raise ContractError('伙伴预算无效')
        if not isinstance(context.get('evidence_ids'), list) or any(not isinstance(e, str) or not e for e in context['evidence_ids']):
            raise ContractError('案件必须登记自己的证据ID，不能由请求临时声明')
        self.directory = Path(directory)
        self.tools = dict(tools)
        self.context = {'version': VERSION, 'context': context, 'tools': sorted(tools),
                        'max_attempts': max_attempts, 'max_rounds': max_rounds}

    def request(self, tool, arguments, *, round_index, evidence_ids, known_evidence, revision=0):
        if tool not in self.tools or type(round_index) is not int or not 1 <= round_index <= self.context['max_rounds']:
            raise ContractError('工具或轮次越界')
        if set(arguments) != {'sample_id'} or arguments['sample_id'] != self.context['context']['sample_id']:
            raise ContractError('伙伴工具参数或案件不合法')
        registered = set(self.context['context']['evidence_ids'])
        if set(known_evidence) != registered or not set(evidence_ids).issubset(registered) or not evidence_ids or revision != self.context['context'].get('decision_revision', 0):
            raise ContractError('伙伴引用缺失、跨案件或修订过期')
        request = {'tool': tool, 'arguments': arguments, 'evidence_ids': sorted(evidence_ids), 'revision': revision}
        rid = digest_json(request)
        with run_lock(self.directory / 'coordinator.lock'):
            path = self.directory / 'state.json'
            state = read_json(path) if path.exists() else {'context': self.context, 'attempts': 0, 'requests': {}}
            if state.get('context') != self.context:
                raise ContractError('伙伴运行版本变化，必须新建目录')
            item = state['requests'].get(rid, {'attempts': 0})
            if item.get('status') == 'ok':
                return item['response']
            if state['attempts'] >= self.context['max_attempts'] or item['attempts'] >= 2:
                raise ContractError('伙伴尝试预算耗尽')
            state['attempts'] += 1
            item = {**item, 'attempts': item['attempts'] + 1, 'status': 'in_progress',
                    'request': request, 'round_index': round_index, 'started_at': time.time()}
            state['requests'][rid] = item
            atomic_json(path, state)
            try:
                response = self.tools[tool](dict(arguments))
                if not isinstance(response, dict) or response.get('status') not in {'ok', 'partial', 'unavailable', 'invalid'}:
                    raise ContractError('伙伴工具响应无效')
                result = response.get('result')
                if response['status'] in {'ok', 'partial'} and not isinstance(result, dict):
                    raise ContractError('成功或部分成功响应必须提供结果映射')
                if isinstance(result, dict) and result.get('sample_id', arguments['sample_id']) != arguments['sample_id']:
                    raise ContractError('伙伴工具返回其他案件')
                digest_json(response)
                item.update(status='ok', response=response)
            except BaseException as exc:
                item.update(status='failed', error_type=type(exc).__name__)
                raise
            finally:
                item['finished_at'] = time.time()
                atomic_json(path, state)
            return response
