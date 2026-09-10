"""只计算明确网格内的候选统计；保留尺度、构件范围和量测未知。"""
import math
from report_contract import ContractError


def extent_from_candidate(candidate, evidence_id):
    result = {'status': 'unknown', 'evidence_ids': [evidence_id], 'physical_width_mm': None,
              'physical_area_m2': None, 'component_damage_ratio': None,
              'grading_basis': 'predicted_unquantified', 'candidate_grid_fraction': None}
    pixels = candidate.get('mask_grid_area_pixels')
    grid = candidate.get('mask_grid_size_hw')
    if pixels is None or grid is None:
        return result
    if not isinstance(grid, list) or len(grid) != 2 or any(type(v) is not int or v <= 0 for v in grid):
        raise ContractError('掩码网格缺失或无效')
    denominator = grid[0] * grid[1]
    if isinstance(pixels, bool) or not isinstance(pixels, (int, float)) or not math.isfinite(pixels) or not 0 <= pixels <= denominator:
        raise ContractError('候选掩码面积越界')
    result['status'] = 'estimated'
    result['candidate_grid_fraction'] = {'value': pixels / denominator, 'unit': 'fraction',
        'denominator': denominator, 'scope': '声明的候选掩码网格', 'measurement_level': 'candidate_statistic'}
    return result
