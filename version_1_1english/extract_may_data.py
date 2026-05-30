#!/usr/bin/env python3
"""
Helper script to extract May 2025 data from Excel file
帮助用户从Excel文件中提取5月份数据
"""

import pandas as pd
import os
from datetime import time

def convert_numeric_time(val):
    """处理时间格式（例如 635 => 06:35）"""
    try:
        if pd.isna(val):
            return None
        val = int(val)
        hour = val // 100
        minute = val % 100
        return time(hour=hour, minute=minute)
    except:
        return None

def extract_may_2025_data(excel_path: str, output_csv: str = "may_2025_actual_data.csv"):
    """
    从Excel文件中提取2025年5月的拥堵数据
    """
    print(f"📊 Extracting May 2025 data from: {excel_path}")
    
    try:
        # 读取Excel文件
        print("🔄 Reading Excel file...")
        df = pd.read_excel(excel_path)
        print(f"✅ Loaded {len(df)} total records")
        
        # 显示列名
        print("\\n📋 Available columns:")
        for i, col in enumerate(df.columns):
            print(f"   {i+1}. {col}")
        
        # 筛选条件
        print("\\n🔍 Applying filters...")
        
        # 1. 筛选5月份数据
        if '月' in df.columns:
            may_data = df[df['月'] == 5]
            print(f"   Month filter: {len(may_data)} records in May")
        else:
            print("❌ Column '月' (Month) not found")
            return False
        
        # 2. 筛选交通集中原因
        if '原因' in df.columns:
            may_data = may_data[may_data['原因'] == '交通集中']
            print(f"   Cause filter: {len(may_data)} traffic concentration events")
        else:
            print("⚠️ Column '原因' (Cause) not found, skipping cause filter")
        
        # 3. 筛选関越道
        if '道路番号' in df.columns:
            kan_etsu_data = may_data[may_data['道路番号'] == '関越道']
            print(f"   Road filter: {len(kan_etsu_data)} events on Kan-Etsu Expressway")
            
            if len(kan_etsu_data) == 0:
                print("⚠️ No Kan-Etsu Expressway data found. Available roads:")
                available_roads = may_data['道路番号'].unique()
                for road in available_roads:
                    count = len(may_data[may_data['道路番号'] == road])
                    print(f"      {road}: {count} events")
                
                # 使用所有可用道路的数据
                print("\\n🔄 Using all available road data for validation...")
                final_data = may_data
            else:
                final_data = kan_etsu_data
        else:
            print("⚠️ Column '道路番号' (Road) not found, using all data")
            final_data = may_data
        
        if len(final_data) == 0:
            print("❌ No data matches the filter criteria")
            return False
        
        # 数据处理
        print("\\n🔧 Processing data...")
        processed_data = final_data.copy()
        
        # 处理时间格式
        if '発生時刻' in processed_data.columns:
            processed_data['発生時刻'] = processed_data['発生時刻'].apply(convert_numeric_time)
        if 'ピーク時刻' in processed_data.columns:
            processed_data['ピーク時刻'] = processed_data['ピーク時刻'].apply(convert_numeric_time)
        
        # 处理数值字段（保留一位小数）
        numeric_cols = ['ピーク長', '発生Ｋｐ', '発生時渋滞長']
        for col in numeric_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                processed_data[col] = processed_data[col].div(10).round(1)
        
        # 创建日期列
        if all(col in processed_data.columns for col in ['年', '月', '日']):
            processed_data['date'] = pd.to_datetime(
                processed_data['年'].astype(str) + '-' +
                processed_data['月'].astype(str).str.zfill(2) + '-' +
                processed_data['日'].astype(str).str.zfill(2),
                errors='coerce'
            )
        
        # 选择需要的列
        required_cols = ['date', '原因', '道路番号', '発生時刻', 'ピーク時刻', 
                        'ピーク長', '発生Ｋｐ', '発生時渋滞長', '渋滞時間']
        available_cols = [col for col in required_cols if col in processed_data.columns]
        
        if 'date' not in available_cols:
            # 如果没有date列，创建一个简单的
            processed_data['date'] = '2025-05-01'  # 默认日期
            available_cols = ['date'] + [col for col in available_cols if col != 'date']
        
        export_data = processed_data[available_cols]
        
        # 保存数据
        print(f"\\n💾 Saving processed data to: {output_csv}")
        export_data.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"✅ Successfully extracted {len(export_data)} records")
        print("\\n📋 Sample of extracted data:")
        print(export_data.head())
        
        print(f"\\n🎯 Data extraction completed! File saved as: {output_csv}")
        print("\\n📊 Data statistics:")
        print(f"   Total events: {len(export_data)}")
        if '道路番号' in export_data.columns:
            road_stats = export_data['道路番号'].value_counts()
            print("   Events by road:")
            for road, count in road_stats.items():
                print(f"      {road}: {count} events")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing Excel file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("📂 May 2025 Data Extraction Tool")
    print("=" * 60)
    
    # 查找Excel文件
    data_dir = "../data/meta_data"
    excel_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if "2025" in file and file.endswith('.xlsx'):
                excel_files.append(os.path.join(data_dir, file))
    
    if not excel_files:
        print("❌ No 2025 Excel files found in ../data/meta_data/")
        print("💡 Please specify the path to your Excel file:")
        excel_path = input("Excel file path: ").strip()
        if not os.path.exists(excel_path):
            print("❌ File not found")
            return False
    else:
        print("📁 Found Excel files:")
        for i, file in enumerate(excel_files):
            print(f"   {i+1}. {os.path.basename(file)}")
        
        if len(excel_files) == 1:
            excel_path = excel_files[0]
            print(f"\\n🎯 Using: {os.path.basename(excel_path)}")
        else:
            try:
                choice = int(input("\\nSelect file (enter number): ")) - 1
                excel_path = excel_files[choice]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return False
    
    # 提取数据
    success = extract_may_2025_data(excel_path)
    
    if success:
        print("\\n🎊 Data extraction successful!")
        print("✅ You can now run: python run_validation.py")
    else:
        print("\\n❌ Data extraction failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()