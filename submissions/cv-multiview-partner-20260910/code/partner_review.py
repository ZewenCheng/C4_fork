"""真实伙伴请求与完整改判事务；缺少冻结采用证据时维持基线。"""
from pathlib import Path
import copy
import json
import time
from report_contract import (ContractError, atomic_json, digest_json, digest_file, read_json, run_lock,
    locked_diagnosis, evidence_catalog, validate_prediction_source, validate_final, assemble_final,
    grade_messages, parse_grade)
from report_contract import QUERY_MAP
from partner_protocol import packet_ledger
from partner_coordinator import PartnerCoordinator


def parse_review(raw, labels, evidence_ids, tools, allow_request):
    if isinstance(raw,str):
        try:raw=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
        except ValueError as exc:raise ContractError('伙伴回复不是JSON') from exc
    required={'action','label','support','opposition','tool','reason'}
    if not isinstance(raw,dict) or set(raw)!=required:raise ContractError('伙伴回复字段不符合合同')
    if raw['action'] not in {'maintain','propose','insufficient','request'} or raw['label'] not in labels:
        raise ContractError('伙伴动作或标签非法')
    for key in ['support','opposition']:
        if not isinstance(raw[key],list) or any(not isinstance(e,str) or e not in evidence_ids for e in raw[key]):
            raise ContractError('伙伴证据引用非法')
    if not isinstance(raw['reason'],str) or not raw['reason'].strip() or len(raw['reason'])>1000:
        raise ContractError('伙伴缺少简短理由')
    if raw['action'] in {'request','propose'} and not raw['support']:raise ContractError('伙伴补证或改判缺少引用')
    if raw['action']=='request':
        if not allow_request or raw['tool'] not in tools:raise ContractError('伙伴请求越界')
    elif raw['tool'] is not None:raise ContractError('非请求回复不得携带工具')
    return raw


def review_messages(baseline, observations, labels, tools, previous=None):
    system=('你是巡检伙伴协调者。依据给定候选指出具体分歧，可请求一个工具补证，或维持、建议改判、证据不足。'
            '原生分数未校准，不可跨模型比较高低；相关模型不算独立票。不得臆造标签、测量或证据。'
            '只返回JSON六字段：action(maintain/propose/insufficient/request)、label、support(证据ID数组)、'
            'opposition(证据ID数组)、tool(请求工具名或null)、reason(简短中文)。'
            'request和propose的support必须包含至少一个observations对象的实际键，禁止空数组。'
            'reason最多40个汉字。第二轮只作结论，不再请求。建议不会直接改变正式答案。')
    payload={'baseline':baseline,'labels':sorted(labels),'tools':sorted(tools) if previous is None else [],
             'observations':observations,'previous':previous}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]


def commit_revision(row, packet, baseline, proposal, catalog, policy, refresh, grader):
    """采用条件必须有冻结验证范围；刷新失败整案返回基线，不输出半个新分支。"""
    if proposal['action']!='propose' or proposal['label']==baseline['defectType']:
        return baseline, {'changed':False,'reason':'维持或没有新类型'}
    allowed=policy.get('validated_transitions',[])
    transition=[baseline['questionCategory'],baseline['defectType'],proposal['label']]
    if not policy.get('validation_sha256') or transition not in allowed:
        return baseline, {'changed':False,'reason':'该类型与来源组合尚无冻结开发采用证据'}
    supporting=[catalog[e] for e in proposal['support'] if e in catalog]
    validated_ids=set(policy.get('domain_validated_evidence_ids',[]))
    if not any(e in validated_ids and catalog[e].get('label')==proposal['label'] for e in proposal['support'] if e in catalog):
        return baseline, {'changed':False,'reason':'缺少已验证领域观测支持'}
    try:
        if not callable(refresh):raise ContractError('缺少可调用的依赖刷新器')
        changed=copy.deepcopy(packet);changed['predicted_type']=proposal['label']
        diagnosis=locked_diagnosis(row,changed)
        diagnosis['source']='partner_decision/revision1'
        refreshed=refresh(diagnosis,proposal)
        if refreshed.get('decision_revision')!=1 or refreshed.get('defectType')!=proposal['label']:
            raise ContractError('新分支依赖未刷新')
        new_catalog=refreshed['catalog']
        # 程度/规范/历史均由刷新器重建或显式移除，不接受旧类型依赖混入。
        for item in new_catalog.values():
            if item.get('dependency_type') not in (None,proposal['label']):raise ContractError('旧类型依赖残留')
        if diagnosis['is_intact']:
            from partner_dependency_refresh import intact_revision_grade
            grade=intact_revision_grade(diagnosis,new_catalog)
        else:
            if not callable(grader):raise ContractError('缺少新类型评级器')
            grade=parse_grade(grader(grade_messages(diagnosis,new_catalog),1),diagnosis,new_catalog)
        final=assemble_final(diagnosis,grade,new_catalog);validate_final(final,diagnosis)
        return final,{'changed':True,'decision_revision':1,'dependencies_sha256':digest_json(refreshed),
                      'grade':grade,'source':'validated_partner_transaction'}
    except (ContractError,RuntimeError,ValueError,KeyError) as exc:
        return baseline,{'changed':False,'reason':'新分支失败，整案回退','error_type':type(exc).__name__}


def run_review(row,packet,baseline,directory,labels,tools,reviewer,*,producer,policy=None,refresh=None,grader=None):
    directory=Path(directory);policy=policy or {'validated_transitions':[],'validation_sha256':None}
    diagnosis=locked_diagnosis(row,packet);validate_prediction_source(diagnosis,evidence_catalog(packet));validate_final(baseline,diagnosis)
    if digest_file(row['path'])!=row['image_sha256']:raise ContractError('伙伴输入内容变化')
    ledger=packet_ledger(row,packet)
    # 明确记录上下文裁剪：每个来源分支只送前两条，实际传入清单独立于完整账本。
    counts={};observations={}
    for o in ledger['observations']:
        key=(o['tool'],o['variant']);counts[key]=counts.get(key,0)+1
        if counts[key]<=2:observations[o['evidence_id']]={'tool':o['tool'],'source_family':o['source_family'],**o['candidate']}
    identity={'input':row['image_sha256'],'packet':digest_json(packet),'baseline':digest_json(baseline),
              'producer':producer,'policy':policy,'labels':sorted(labels),'tools':sorted(tools),
              'implementation':digest_file(__file__)}
    identity['dependencies']={n:digest_file(Path(__file__).with_name(n)) for n in
        ['partner_protocol.py','partner_coordinator.py','report_contract.py','metadata_semantics.py','partner_dependency_refresh.py']}
    identity['refresh']=getattr(refresh,'version',None)
    with run_lock(directory/'review.lock'):
        target=directory/'complete.json'
        if target.exists():
            saved=read_json(target)
            if saved['identity']!=identity or digest_json(saved['result'])!=saved['result_sha256']:
                raise ContractError('伙伴恢复凭证或版本不一致')
            return saved
        state_path=directory/'state.json'
        state=read_json(state_path) if state_path.exists() else {'identity':identity,'calls':0,'turns':[]}
        if state['identity']!=identity:raise ContractError('伙伴恢复上下文变化')
        coordinator=PartnerCoordinator(directory/'tools',{'sample_id':row['sample_id'],'decision_revision':0,
            'evidence_ids':sorted(observations)},tools)
        decision=state.get('decision')
        # 恢复已完成工具结果，模型最多两次；中断调用消耗额度，不重置。
        if state.get('observations'):observations=state['observations']
        while state['calls']<4 and decision is None:
            previous=state['turns'][-1] if state['turns'] else None
            wire_ids={'e'+digest_json(e)[:10]:e for e in observations}
            if len(wire_ids)!=len(observations):raise ContractError('短证据标识冲突')
            wire_observations={short:observations[full] for short,full in wire_ids.items()}
            # 上轮仅给动作及错误，不把内部长ID或原始冗长回复反复塞回提示。
            prior_summary=None if previous is None else {'action':previous.get('response',{}).get('action'),'error':previous.get('error')}
            messages=review_messages(diagnosis,wire_observations,labels,tools,prior_summary)
            requested=any(t.get('response',{}).get('action')=='request' for t in state['turns'])
            stage_attempts=sum(t.get('stage')==('conclusion' if requested else 'request') for t in state['turns'])
            if stage_attempts>=2:
                break
            if previous and previous.get('error'):
                messages[0]['content']+='上次格式未通过：'+previous['error']+'。修正字段与引用，保持JSON合同。'
                payload=json.loads(messages[1]['content']);payload['tools']=[] if requested else sorted(tools)
                messages[1]['content']=json.dumps(payload,ensure_ascii=False)
            prepare=getattr(reviewer,'prepare_messages',None)
            if callable(prepare):messages=prepare(messages)
            state['calls']+=1;atomic_json(state_path,state)
            turn={'round':2 if requested else 1,'stage':'conclusion' if requested else 'request',
                  'request_sha256':digest_json(messages),'sent_evidence_ids':sorted(observations),'wire_to_stable_ids':wire_ids}
            try:
                raw=reviewer(messages,2 if stage_attempts else 1)
                turn['raw_response']=raw
                answer=parse_review(raw,labels,wire_observations,tools,not requested)
                answer=copy.deepcopy(answer)
                for key in ['support','opposition']:answer[key]=[wire_ids[e] for e in answer[key]]
                turn['response']=answer
                if answer['action']=='request':
                    response=coordinator.request(answer['tool'],{'sample_id':row['sample_id']},round_index=1,
                        evidence_ids=answer['support'],known_evidence=coordinator.context['context']['evidence_ids'])
                    turn['tool_response']=response
                    if response['status'] in {'ok','partial'}:
                        extra=packet_ledger(row,{'sample_id':row['sample_id'],'evidence':{answer['tool']:response}})
                        sent=[];groups={}
                        target_labels={diagnosis['defectType'],answer['label']}
                        target_queries={q for term,q in QUERY_MAP if any(term in label for label in target_labels)}
                        for o in extra['observations']:groups.setdefault(o['variant'],[]).append(o)
                        selected=[]
                        for group in groups.values():
                            selected.extend(sorted(group,key=lambda o:(not (o['candidate'].get('label') in target_labels or
                                o['candidate'].get('query') in target_queries),-o['candidate']['uncalibrated_score']))[:5])
                        for o in selected:
                            observations[o['evidence_id']]={'tool':o['tool'],'source_family':o['source_family'],**o['candidate']}
                            sent.append(o['evidence_id'])
                        turn['handoff']={'received':len(extra['observations']),'sent_evidence_ids':sent,
                            'not_sent_ids':[o['evidence_id'] for o in extra['observations'] if o['evidence_id'] not in sent],
                            'reason':'当前与争议类型对应候选优先，每分支最多5条；完整回应保留，未传入项不算模型已读'}
                else:decision=answer
            except (ContractError,RuntimeError,ValueError) as exc:
                turn['error_type']=type(exc).__name__
                turn['error']=str(exc)[:500]
                if not isinstance(exc,ContractError) or turn.get('response'):
                    decision={'action':'insufficient','label':diagnosis['defectType'],'support':[], 'opposition':[], 'tool':None,'reason':'补证或协调失败'}
            state['turns'].append(turn);state['observations']=observations;state['decision']=decision;atomic_json(state_path,state)
        if decision is None:decision={'action':'insufficient','label':diagnosis['defectType'],'support':[],'opposition':[], 'tool':None,'reason':'调用额度已用尽'}
        def bounded_grade(messages,attempt):
            if not callable(grader):raise ContractError('缺少新类型评级器')
            signature=digest_json(messages)
            saved=state.get('revision_grade')
            if saved is not None:
                if saved['request_sha256']!=signature:raise ContractError('评级恢复上下文变化')
                return saved['raw_response']
            if state['calls']>=4:raise ContractError('伙伴改判评级的共享调用额度已用尽')
            state['calls']+=1;atomic_json(state_path,state)
            raw=grader(messages,attempt)
            state['revision_grade']={'request_sha256':signature,'raw_response':raw}
            atomic_json(state_path,state)
            return raw
        final,transaction=commit_revision(row,packet,baseline,decision,observations,policy,refresh,bounded_grade)
        result={'identity':identity,'decision':decision,'transaction':transaction,'result':final,
                'result_sha256':digest_json(final),'review_calls':state['calls'],'turns':state['turns'],'revision_grade':state.get('revision_grade'),
                'not_sent_initial_evidence_ids':[o['evidence_id'] for o in ledger['observations'] if o['evidence_id'] not in observations]}
        atomic_json(target,result)
        return result
