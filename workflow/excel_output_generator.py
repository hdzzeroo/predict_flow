"""
Excel格式输出生成器
将现有系统的聚类结果转换为渋滞予測フォーマット格式
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os

class ExcelOutputGenerator:
    
    def __init__(self):
        """初始化Excel输出生成器"""
        self.road_data = {}  # 道路信息缓存
        self.road_name_map = {
            '関越道': '関越自動車道',
            '東北道': '東北自動車道',
            '中央道': '中央自動車道'
        }
        self.road_code_map = {
            '関越道': '1800',
            '東北道': '1040',
            '中央道': '1300'
        }

    def _convert_hotspot_to_hull(self, hotspot: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """
        将LLM的hotspot转换为hull格式

        Args:
            hotspot: LLM识别的热点数据
            direction: 方向

        Returns:
            hull格式的数据字典
        """
        # 提取hull_triangle坐标
        hull_triangle = hotspot.get('hull_triangle', {})

        # 如果没有hull_triangle，从kp_range和time_range计算
        if not hull_triangle:
            kp_range = hotspot['kp_range']
            time_range = hotspot['time_range']

            # 估算中心位置
            kp_center = (kp_range[0] + kp_range[1]) / 2
            time_center = (time_range[0] + time_range[1]) / 2

            hull_triangle = {
                'kp_start': kp_range[0],
                'kp_end': kp_range[1],
                'peak_kp': kp_center,
                'time_start': time_range[0],
                'time_end': time_range[1],
                'peak_time': time_center
            }

        # 构造hull字典（兼容旧格式）
        return {
            'cluster_id': hotspot['hotspot_id'],
            'cluster_size': hotspot['frequency'],
            'kp_center': hull_triangle['peak_kp'],
            'time_center': hull_triangle['peak_time'],
            'kp_range': hotspot['kp_range'],
            'time_range': hotspot['time_range'],
            'center': [hull_triangle['peak_kp'], hull_triangle['peak_time']],
            'direction': direction,
            'source': 'llm'
        }
        
    def load_road_info(self, road_type: str) -> pd.DataFrame:
        """加载道路信息CSV文件"""
        if road_type in self.road_data:
            return self.road_data[road_type]
            
        csv_file_map = {
            '関越道': '/home/dizhihuang/graduate/predict_workflow/data/roadic_kannetsu.csv',
            '東北道': '/home/dizhihuang/graduate/predict_workflow/data/roadic_touhoku.csv'
        }
        
        if road_type not in csv_file_map:
            raise ValueError(f"不支持的道路类型: {road_type}")
            
        try:
            df = pd.read_csv(csv_file_map[road_type])
            self.road_data[road_type] = df
            return df
        except FileNotFoundError:
            print(f"警告: 道路信息文件未找到: {csv_file_map[road_type]}")
            return pd.DataFrame()
    
    def find_road_section(self, road_type: str, kp: float, direction: str) -> Dict[str, str]:
        """根据KP和方向查找道路区间信息"""
        road_df = self.load_road_info(road_type)
        if road_df.empty:
            return {"start_name": "Unknown", "end_name": "Unknown", "section": "Unknown"}
            
        # 过滤方向
        direction_filter = 'up' if direction == '上' else 'down'
        filtered_df = road_df[road_df['direction'] == direction_filter].copy()
        
        if filtered_df.empty:
            return {"start_name": "Unknown", "end_name": "Unknown", "section": "Unknown"}
        
        # 找到最接近的KP点
        filtered_df.loc[:, 'kp_diff'] = abs(filtered_df['KP'] - kp)
        closest_row = filtered_df.loc[filtered_df['kp_diff'].idxmin()]
        
        return {
            "start_name": closest_row['start_name'],
            "end_name": closest_row['end_name'],
            "section": f"{closest_row['start_name']}～{closest_row['end_name']}"
        }
    
    def convert_time_to_range(self, center_time_minutes: float) -> tuple:
        """将中心时间转换为时间段"""
        # 将分钟转换为小时:分钟格式
        hours = int(center_time_minutes // 60)
        minutes = int(center_time_minutes % 60)
        center_time = f"{hours:02d}:{minutes:02d}"
        
        # 创建时间段（前后各扩展1-2小时）
        start_hour = max(0, hours - 2)
        end_hour = min(23, hours + 2)
        
        time_range = f"{start_hour:02d}:00～{end_hour:02d}:00"
        peak_time = f"{hours}時"
        
        return time_range, peak_time
    
    def generate_time_header_row(self) -> List[str]:
        """生成72小时时间段的标题行"""
        time_headers = []
        
        # 前15个固定列的标题
        fixed_headers = [
            '道路\nコード', '道路名', '月日', '方向', '区間（自）', '区間（至）', 
            'ボトルネック箇所', 'KP', '時間帯', 'ピーク\n時間', 'ピーク時\nの渋滞長',
            '通常\n所要\n時間', '渋滞\n所要\n時間', '増加\n所要\n時間', '発生日'
        ]
        time_headers.extend(fixed_headers)
        
        # 添加72小时的时间标题 (0时到71时，对应3天)
        for hour in range(72):
            day = hour // 24 + 1  # 第几天 (1, 2, 3)
            hour_in_day = hour % 24  # 当天的小时数
            time_headers.append(f"{hour_in_day}")
        
        # 最后3列
        time_headers.extend(['社名', 'データ更新', '路線跨り'])
        
        return time_headers
    
    def calculate_congestion_pattern(self, hull: Dict[str, Any], peak_congestion_km: float = 5.0) -> List[str]:
        """
        根据三角形形状计算拥堵长度的时间分布模式
        
        Args:
            hull: 外包大三角形数据
            peak_congestion_km: 峰值拥堵长度(km)
            
        Returns:
            72个小时的拥堵长度列表
        """
        # 获取时间范围信息
        time_center = hull.get('time_center', hull.get('center', [0, 0])[1])
        if isinstance(time_center, list):
            time_center = time_center[1]
        
        # 获取三角形的时间范围
        time_range = hull.get('time_range', [time_center - 120, time_center + 120])  # 默认前后2小时
        start_time_minutes = time_range[0]
        end_time_minutes = time_range[1]
        peak_time_minutes = time_center
        
        # 转换为小时
        start_hour = max(0, int(start_time_minutes // 60))
        end_hour = min(71, int(end_time_minutes // 60))
        peak_hour = int(peak_time_minutes // 60)
        
        # 初始化72小时数据
        congestion_data = [''] * 72
        
        # 确保时间范围合理
        if start_hour >= 72 or end_hour < 0 or start_hour >= end_hour:
            return congestion_data
        
        # 计算拥堵分布模式（三角形分布）
        duration = end_hour - start_hour + 1
        
        if duration <= 1:
            # 持续时间很短，只在峰值时间填充
            if 0 <= peak_hour < 72:
                congestion_data[peak_hour] = str(int(peak_congestion_km))
        else:
            # 计算三角形分布
            for hour in range(start_hour, min(end_hour + 1, 72)):
                if hour == peak_hour:
                    # 峰值时间
                    congestion_data[hour] = str(int(peak_congestion_km))
                else:
                    # 计算距离峰值的比例
                    if hour < peak_hour:
                        # 上升阶段
                        distance_ratio = (hour - start_hour + 1) / (peak_hour - start_hour + 1) if peak_hour > start_hour else 1
                    else:
                        # 下降阶段  
                        distance_ratio = (end_hour - hour + 1) / (end_hour - peak_hour + 1) if end_hour > peak_hour else 1
                    
                    # 计算拥堵长度（三角形分布，边缘趋向于0）
                    congestion_length = peak_congestion_km * distance_ratio
                    
                    # 设置最小阈值，小于1km的设为0，明确表示无拥堵
                    if congestion_length >= 1.0:
                        congestion_data[hour] = str(int(congestion_length))
                    else:
                        congestion_data[hour] = '0'
        
        # 确保开始和结尾时间为0（明确表示拥堵开始和结束）
        if start_hour < 72:
            congestion_data[start_hour] = '0'
        if end_hour < 72:
            congestion_data[end_hour] = '0'
        
        return congestion_data
    
    def generate_bottleneck_description(self, road_section: Dict[str, str], kp: float) -> str:
        """生成瓶颈位置描述"""
        if road_section['end_name'] != "Unknown":
            return f"{road_section['end_name']}付近"
        else:
            return f"KP{kp:.1f}km付近"
    
    def convert_workflow_to_csv_format(
        self,
        workflow_result: Dict[str, Any],
        target_date: str = None
    ) -> pd.DataFrame:
        """将工作流结果转换为标准CSV格式"""

        if target_date is None:
            # 尝试从workflow_result中获取目标日期
            target_month = workflow_result.get('target_month')
            target_day = workflow_result.get('target_day')
            if target_month and target_day:
                target_date = f"{target_month:02d}月{target_day:02d}日"
            else:
                # 如果没有提供目标日期，使用当前日期
                target_date = datetime.now().strftime("%m月%d日")
        
        # 提取基本信息
        route = workflow_result.get('route', '関越道')
        # 如果route为None或空，使用默认值
        if not route or route == "未指定" or route == "Not specified":
            route = '関越道'
        road_name = self.road_name_map.get(route, route)
        road_code = self.road_code_map.get(route, '1800')
        
        csv_rows = []

        # 检查是否有LLM分析结果
        llm_analysis = workflow_result.get('llm_analysis', {})

        # 处理方向数据
        direction_data = workflow_result.get('direction_data', {})

        if not direction_data:
            # 兼容旧格式
            direction_data = {'未指定': {
                'hulls': workflow_result.get('hulls', []),
                'triangles': workflow_result.get('triangles', [])
            }}

        for direction_key, data in direction_data.items():
            direction = direction_key if direction_key in ['上', '下'] else '上'

            # 优先使用LLM分析结果
            hulls_to_process = []

            if llm_analysis and direction_key in llm_analysis:
                # 使用LLM的hotspots，转换为hull格式
                hotspots = llm_analysis[direction_key].get('hotspots', [])

                for hotspot in hotspots:
                    hull = self._convert_hotspot_to_hull(hotspot, direction)
                    hulls_to_process.append(hull)
            else:
                # 使用传统的hulls
                hulls_to_process = data.get('hulls', [])

            if not hulls_to_process:
                continue

            for hull in hulls_to_process:
                # 获取聚类中心位置和时间信息
                kp_center = hull.get('kp_center', hull.get('center', [0, 0])[0])
                time_center = hull.get('time_center', hull.get('center', [0, 0])[1])
                
                if isinstance(kp_center, list):
                    kp_center = kp_center[0]
                if isinstance(time_center, list):
                    time_center = time_center[1]
                
                # 查找道路区间信息
                road_section = self.find_road_section(route, kp_center, direction)
                
                # 转换时间信息
                time_range, peak_time = self.convert_time_to_range(time_center)
                
                # 生成瓶颈描述
                bottleneck = self.generate_bottleneck_description(road_section, kp_center)
                
                # 计算峰值拥堵长度（从hull数据中获取或使用默认值）
                peak_congestion_km = 5.0  # 默认5km
                if 'cluster_size' in hull:
                    # 根据聚类大小调整峰值拥堵长度
                    cluster_size = hull['cluster_size']
                    peak_congestion_km = min(10.0, max(2.0, cluster_size * 1.5))
                
                # 根据三角形形状计算72小时拥堵分布
                time_data = self.calculate_congestion_pattern(hull, peak_congestion_km)
                
                # 创建完整的CSV行记录，严格按照模板格式
                csv_row = []
                
                # 构建完整的90列数据
                csv_row.extend([
                    road_code,                           # 1. 道路コード
                    road_name,                           # 2. 道路名  
                    target_date,                         # 3. 月日
                    f"{direction}り",                     # 4. 方向
                    road_section['start_name'],          # 5. 区間（自）
                    road_section['end_name'],            # 6. 区間（至）
                    bottleneck,                          # 7. ボトルネック箇所
                    f"{kp_center:.1f} ",                 # 8. KP (加空格匹配模板)
                    time_range,                          # 9. 時間帯
                    peak_time,                           # 10. ピーク時間
                    f"{int(peak_congestion_km)}km",      # 11. ピーク時の渋滞長 (根据聚类计算)
                    "00:04",                             # 12. 通常所要時間 (模板值)
                    "00:15",                             # 13. 渋滞所要時間 (模板值)
                    "00:11",                             # 14. 増加所要時間 (模板值)
                    ""                                   # 15. 発生日 (空列)
                ])
                
                # 第16-87列：72小时时间段数据（使用计算的拥堵分布模式）
                csv_row.extend(time_data)
                
                # 最后3列：元数据 (第88-90列)
                csv_row.extend([
                    "東日本",      # 88. 社名
                    "",            # 89. データ更新  
                    ""             # 90. 路線跨り
                    # 注意：支社名列已经不在这里了，模板只有90列
                ])
                
                csv_rows.append(csv_row)
        
        # 创建DataFrame，使用标准列名（确保90列）
        if csv_rows:
            # 检查数据列数
            if csv_rows and len(csv_rows[0]) != 90:
                print(f"警告: 数据列数不匹配，期望90列，实际{len(csv_rows[0])}列")
                
            # 定义完整的90列列名
            columns = [
                '道路\nコード', '道路名', '月日', '方向', '区間（自）', '区間（至）', 
                'ボトルネック箇所', 'KP', '時間帯', 'ピーク\n時間', 'ピーク時\nの渋滞長',
                '通常\n所要\n時間', '渋滞\n所要\n時間', '増加\n所要\n時間', '発生日'
            ]
            
            # 添加72小时列名 (空字符串，因为原模板就是这样)
            columns.extend([''] * 72)
            
            # 添加最后3列 (总共90列)
            columns.extend(['社名', 'データ更新', '路線跨り'])
            
            # 确保列名数量正确
            if len(columns) != 90:
                print(f"错误: 列名定义错误，期望90列，实际{len(columns)}列")
                columns = columns[:90] + [''] * max(0, 90 - len(columns))
            
            df = pd.DataFrame(csv_rows, columns=columns)
            return df
        else:
            # 返回空的DataFrame
            return pd.DataFrame()
        
    def convert_workflow_to_excel_format(self, workflow_result: Dict[str, Any], target_date: str = None) -> pd.DataFrame:
        """兼容性方法，调用CSV格式转换"""
        return self.convert_workflow_to_csv_format(workflow_result, target_date)
    
    def save_csv_output(
        self, 
        workflow_result: Dict[str, Any], 
        output_path: str,
        target_date: str = None
    ) -> str:
        """保存CSV格式输出"""
        
        df = self.convert_workflow_to_csv_format(workflow_result, target_date)
        
        if df.empty:
            print("警告: 没有生成任何拥堵预测数据")
            return output_path
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 添加标题行（模仿原始格式）
            title_row = "渋滞予測　令和07年04月01日～令和07年04月30日" + "," * (len(df.columns) - 1)
            
            # 生成时间标题行
            time_headers = self.generate_time_header_row()
            
            # 保存为CSV文件
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                # 写入主标题行
                f.write(title_row + '\n')
                
                # 写入时间标题行
                f.write(','.join([f'"{header}"' for header in time_headers]) + '\n')
                
                # 写入数据（不包含列名，因为已经写入了自定义标题）
                df.to_csv(f, index=False, header=False, quoting=1)  # header=False 不写入默认列名
            
            print(f"CSV输出已保存至: {output_path}")
            print(f"共生成 {len(df)} 条拥堵预测记录")
            return output_path
            
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
            return output_path
    
    def save_excel_output(
        self, 
        workflow_result: Dict[str, Any], 
        output_path: str,
        target_date: str = None
    ) -> str:
        """保存Excel格式输出（兼容性方法）"""
        
        # 如果输出路径是.csv，调用CSV保存方法
        if output_path.endswith('.csv'):
            return self.save_csv_output(workflow_result, output_path, target_date)
        
        df = self.convert_workflow_to_csv_format(workflow_result, target_date)
        
        if df.empty:
            print("警告: 没有生成任何拥堵预测数据")
            return output_path
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存为Excel文件
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='拥堵预测', index=False)
                
                # 获取工作表并设置格式
                worksheet = writer.sheets['拥堵预测']
                
                # 设置列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 15)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"Excel输出已保存至: {output_path}")
            print(f"共生成 {len(df)} 条拥堵预测记录")
            return output_path
            
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")
            return output_path

# 辅助函数
def generate_excel_prediction_output(workflow_result: Dict[str, Any], output_dir: str = None, format_type: str = 'csv') -> str:
    """生成格式化的拥堵预测输出"""

    if output_dir is None:
        # 使用相对路径，自动适配当前目录
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    # 生成文件名 - 改进route获取逻辑
    route = workflow_result.get('route')
    if not route or route == 'Not specified':
        # 尝试从direction_data中获取第一个方向的数据来推断route
        direction_data = workflow_result.get('direction_data', {})
        if direction_data:
            for dir_data in direction_data.values():
                triangles = dir_data.get('triangles', [])
                if triangles and 'road_type' in triangles[0]:
                    route = triangles[0]['road_type']
                    break
        # 如果还是没有，使用默认值
        if not route:
            route = '関越道'

    # 生成时间戳 - 优先使用目标日期
    target_year = workflow_result.get('target_year')
    target_month = workflow_result.get('target_month')
    target_day = workflow_result.get('target_day')

    if target_year and target_month and target_day:
        # 使用预测目标日期
        timestamp = f"{target_year}{target_month:02d}{target_day:02d}_{datetime.now().strftime('%H%M%S')}"
    elif target_year and target_month:
        # 只有年月，使用月初
        timestamp = f"{target_year}{target_month:02d}01_{datetime.now().strftime('%H%M%S')}"
    else:
        # 回退到当前日期
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'csv':
        output_filename = f"congestion_prediction_{route}_{timestamp}.csv"
    else:
        output_filename = f"congestion_prediction_{route}_{timestamp}.xlsx"
        
    output_path = os.path.join(output_dir, output_filename)
    
    # 创建生成器并保存文件
    generator = ExcelOutputGenerator()
    
    if format_type == 'csv':
        return generator.save_csv_output(workflow_result, output_path)
    else:
        return generator.save_excel_output(workflow_result, output_path)

def generate_csv_prediction_output(workflow_result: Dict[str, Any], output_dir: str = None) -> str:
    """生成CSV格式的拥堵预测输出"""
    return generate_excel_prediction_output(workflow_result, output_dir, format_type='csv')

if __name__ == "__main__":
    # 测试代码
    print("Excel输出生成器已准备就绪")
    print("使用方法: generate_excel_prediction_output(workflow_result)")