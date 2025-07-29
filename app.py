import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import platform

# if platform.system() == "Windows":
#     plt.rcParams['font.family'] = ['SimHei'] # Windows
# elif platform.system() == "Darwin":
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['axes.unicode_minus']=False 
# plt.rcParams['font.family'] = ['SimHei'] # Windows
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os

def setup_chinese_font():
    # 下载中文字体文件
    font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SourceHanSansSC-Regular.otf"
    font_path = "SourceHanSansSC-Regular.otf"
    
    if not os.path.exists(font_path):
        try:
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)
        except:
            st.warning("字体下载失败，将使用默认字体")
            return
    
    # 注册字体
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = ['Source Han Sans SC']
    
    plt.rcParams['axes.unicode_minus'] = False

# 在streamlit app开始时调用
setup_chinese_font()

# 设置页面配置
st.set_page_config(
    page_title="糖尿病预测决策树系统",
    page_icon="🌳",
    layout="wide"
)

# 设置工作目录
os.chdir(os.path.split(__file__)[0] if '__file__' in globals() else os.getcwd())

# 全局变量存储编码器
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'target_name' not in st.session_state:
    st.session_state.target_name = ""

def load_and_encode_data(file_path=None, uploaded_file=None):
    """加载和编码数据"""
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            data = pd.read_csv(file_path, encoding='gbk')
        
        # 获取特征名和目标变量名
        feature_names = data.columns[:-1].tolist()
        target_name = data.columns[-1]
        
        # 存储原始特征值用于下拉框显示
        original_values = {}
        encoders = {}
        
        # 编码所有列
        encoded_data = data.copy()
        for column in data.columns:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(data[column])
            encoders[column] = le
            if column != target_name:
                original_values[column] = data[column].unique().tolist()
        
        return encoded_data, encoders, feature_names, target_name, original_values, data
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None, None, None, None, None, None

def calculate_gini(y):
    """计算基尼系数"""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def calculate_feature_gini(X, y, feature_idx):
    """计算特征的基尼系数"""
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values)
    
    weighted_gini = 0
    total_samples = len(y)
    
    for value in unique_values:
        mask = feature_values == value
        subset_y = y[mask]
        weight = len(subset_y) / total_samples
        weighted_gini += weight * calculate_gini(subset_y)
    
    return weighted_gini

# 密码验证和页面选择
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.page_type = None

if not st.session_state.authenticated:
    st.title("🌳 糖尿病预测决策树系统")
    st.markdown("---")
    st.subheader("🔐 访问验证")
    
    password = st.text_input("请输入访问密码", type="password")
    
    if st.button("🚀 进入系统", type="primary"):
        if password == "0":
            st.session_state.authenticated = True
            st.session_state.page_type = "quick_prediction"
            st.rerun()
        elif password == "1":
            st.session_state.authenticated = True
            st.session_state.page_type = "gini_analysis"
            st.rerun()
        elif password == "2":
            st.session_state.authenticated = True
            st.session_state.page_type = "performance_analysis"
            st.rerun()
        elif password == "3":
            st.session_state.authenticated = True
            st.session_state.page_type = "complete_training"
            st.rerun()
        else:
            st.error("❌ 密码错误，请重新输入")
    

else:
    # 根据验证的密码显示对应页面
    if st.session_state.page_type == "quick_prediction":
        st.title("🔮 快速体验预测")
        st.markdown("自动优化模型参数并进行糖尿病风险预测")
        
        st.subheader("📁 数据选择")
        
        # 文件上传或使用默认数据
        uploaded_file = st.file_uploader("📁 上传CSV文件", type=['csv'], key="quick_file")
        use_default = st.checkbox("使用默认数据文件 (young-diabetes.csv)", key="quick_default")
        
        # 当有数据输入时自动进行训练
        if uploaded_file is not None or use_default:
            with st.spinner('正在加载数据并寻找最佳参数...'):
                # 加载数据
                if use_default:
                    data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data("young-diabetes.csv")
                else:
                    data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data(uploaded_file=uploaded_file)
                
                if data is not None:
                    # 准备数据
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    
                    # 网格搜索最佳参数（不显示过程）
                    from sklearn.model_selection import GridSearchCV
                    
                    param_grid = {
                        'max_depth': [3, 5, 7, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                    
                    dt = DecisionTreeClassifier(random_state=42)
                    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    grid_search.fit(X, y)
                    
                    # 使用最佳参数和全部数据训练最终模型
                    best_model = grid_search.best_estimator_
                    best_model.fit(X, y)
                    
                    # 保存模型和相关信息
                    model_info = {
                        'model': best_model,
                        'encoders': encoders,
                        'feature_names': feature_names,
                        'target_name': target_name,
                        'original_values': original_values,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_
                    }
                    
                    model_path = "diabetes_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_info, f)
                    
                    st.session_state.quick_model_info = model_info
                    

            
            # 显示预测界面
            st.markdown("---")
            st.subheader("🎯 风险预测")
            
            # 检查是否有训练好的模型
            model_available = False
            
            # 先检查session_state中是否有模型
            if 'quick_model_info' in st.session_state:
                model_info = st.session_state.quick_model_info
                model_available = True
            # 再检查是否有保存的模型文件
            elif os.path.exists("diabetes_model.pkl"):
                try:
                    with open("diabetes_model.pkl", 'rb') as f:
                        model_info = pickle.load(f)
                    model_available = True
                except:
                    model_available = False
            
            if model_available:
                model = model_info['model']
                encoders = model_info['encoders']
                feature_names = model_info['feature_names']
                target_name = model_info['target_name']
                original_values = model_info['original_values']
                
                st.write("请选择各项特征值：")
                
                # 创建输入表单
                input_data = {}
                cols = st.columns(3)  # 创建3列布局
                
                for i, feature in enumerate(feature_names):
                    with cols[i % 3]:
                        if feature in original_values:
                            selected_value = st.selectbox(
                                f"📊 {feature}",
                                original_values[feature],
                                key=f"quick_{feature}"
                            )
                            input_data[feature] = selected_value
                
                if st.button("🔍 开始预测", type="primary"):
                    try:
                        # 编码输入数据
                        encoded_input = []
                        for feature in feature_names:
                            encoded_value = encoders[feature].transform([input_data[feature]])[0]
                            encoded_input.append(encoded_value)
                        
                        # 进行预测
                        prediction = model.predict([encoded_input])[0]
                        prediction_proba = model.predict_proba([encoded_input])[0]
                        
                        # 解码预测结果
                        result = encoders[target_name].inverse_transform([prediction])[0]
                        
                        # 显示结果
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if result == "高风险" or "高" in str(result):
                                st.error(f"⚠️ 预测结果: **{result}**")
                                st.error(f"风险概率: {prediction_proba[prediction]:.2%}")
                            else:
                                st.success(f"✅ 预测结果: **{result}**")
                                st.success(f"安全概率: {prediction_proba[prediction]:.2%}")
                        
                        with col2:
                            # 显示概率分布
                            prob_df = pd.DataFrame({
                                '类别': encoders[target_name].classes_,
                                '概率': prediction_proba
                            })
                            
                            fig = px.bar(prob_df, x='类别', y='概率', 
                                       title="各类别预测概率",
                                       color='概率',
                                       color_continuous_scale='RdYlBu_r')
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"预测失败: {str(e)}")
                        
            else:
                st.warning("⚠️ 模型训练失败，请重新尝试！")
        
        else:
            st.info("📌 请上传CSV文件或选择使用默认数据来开始体验预测功能")

    elif st.session_state.page_type == "complete_training":
        st.title("🛠️ 完整模型训练")
        st.markdown("导入数据并自定义训练决策树模型")
        
        # 文件上传
        uploaded_file = st.file_uploader("📁 上传CSV文件", type=['csv'])
        use_default = st.checkbox("使用默认数据文件 (young-diabetes.csv)")
        
        if uploaded_file is not None or use_default:
            # 加载数据
            if use_default:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data("young-diabetes.csv")
            else:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data(uploaded_file=uploaded_file)
            
            if data is not None:
                # 显示数据信息
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 数据预览")
                    st.write("原始数据:")
                    try:
                        st.dataframe(raw_data.head())
                    except ImportError:
                        # 如果pyarrow有问题，使用表格显示
                        st.table(raw_data.head())
                    

                with col2:
                    st.subheader("📈 数据统计")
                    st.write(f"样本数量: {len(data)}")
                    st.write(f"特征数量: {len(feature_names)}")
                    st.write(f"目标变量: {target_name}")
                    
                    # 目标变量分布
                    target_counts = raw_data[target_name].value_counts()
                    fig = px.pie(values=target_counts.values, names=target_counts.index,
                               title=f"{target_name} 分布")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 模型参数设置
                st.subheader("⚙️ 模型参数设置")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    max_depth = st.slider("最大深度", 1, 20, 5)
                with col2:
                    min_samples_split = st.slider("分裂最小样本", 2, 20, 2)
                with col3:
                    min_samples_leaf = st.slider("叶节点最小样本", 1, 10, 1)
                with col4:
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
                
                if st.button("🚀 训练模型", type="primary"):
                    # 准备数据
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    
                    # 分割数据
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # 训练模型
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    
                    with st.spinner('正在训练模型...'):
                        model.fit(X_train, y_train)
                    
                    # 预测
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # 计算准确率
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    # 显示结果
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 模型性能")
                        st.metric("训练集准确率", f"{train_acc:.3f}")
                        st.metric("测试集准确率", f"{test_acc:.3f}")
                        
                        # 分类报告
                        report = classification_report(y_test, test_pred, 
                                                     target_names=encoders[target_name].classes_,
                                                     output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.write("分类报告:")
                        try:
                            st.dataframe(report_df)
                        except ImportError:
                            # 如果pyarrow有问题，使用表格显示
                            st.table(report_df.round(3))
                    
                    with col2:
                        st.subheader("🌳 决策树可视化")
                        fig, ax = plt.subplots(figsize=(15, 10))
                        plot_tree(model, 
                                 feature_names=feature_names,
                                 class_names=encoders[target_name].classes_,
                                 filled=True, 
                                 rounded=True,
                                 fontsize=8)
                        st.pyplot(fig)
                    
                    # 保存模型
                    st.session_state['trained_model'] = model
                    st.session_state['model_encoders'] = encoders
                    st.session_state['model_features'] = feature_names
                    st.session_state['model_target'] = target_name
                    st.session_state['model_original_values'] = original_values
                    
                    st.success("✅ 模型训练完成！")
                
                # 预测部分
                if 'trained_model' in st.session_state:
                    st.markdown("---")
                    st.subheader("🎯 新数据预测")
                    
                    model = st.session_state['trained_model']
                    encoders = st.session_state['model_encoders']
                    feature_names = st.session_state['model_features']
                    target_name = st.session_state['model_target']
                    original_values = st.session_state['model_original_values']
                    
                    # 创建预测输入
                    cols = st.columns(3)
                    input_data = {}
                    
                    for i, feature in enumerate(feature_names):
                        with cols[i % 3]:
                            if feature in original_values:
                                selected_value = st.selectbox(
                                    f"{feature}",
                                    original_values[feature],
                                    key=f"pred_{feature}"
                                )
                                input_data[feature] = selected_value
                    
                    if st.button("🔍 预测", type="primary"):
                        try:
                            # 编码输入数据
                            encoded_input = []
                            for feature in feature_names:
                                encoded_value = encoders[feature].transform([input_data[feature]])[0]
                                encoded_input.append(encoded_value)
                            
                            # 进行预测
                            prediction = model.predict([encoded_input])[0]
                            prediction_proba = model.predict_proba([encoded_input])[0]
                            
                            # 解码预测结果
                            result = encoders[target_name].inverse_transform([prediction])[0]
                            
                            # 显示结果
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if "高风险" in str(result) or "高" in str(result):
                                    st.error(f"⚠️ 预测结果: **{result}**")
                                else:
                                    st.success(f"✅ 预测结果: **{result}**")
                            
                            with col2:
                                # 概率分布图
                                prob_df = pd.DataFrame({
                                    '类别': encoders[target_name].classes_,
                                    '概率': prediction_proba
                                })
                                
                                fig = px.bar(prob_df, x='类别', y='概率',
                                           title="预测概率分布",
                                           color='概率',
                                           color_continuous_scale='RdYlBu_r')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"预测失败: {str(e)}")

    elif st.session_state.page_type == "gini_analysis":
        st.title("📊 基尼系数分析")
        st.markdown("分析各特征的基尼不纯度系数")
        
        # 文件上传
        uploaded_file = st.file_uploader("📁 上传CSV文件", type=['csv'], key="gini_file")
        use_default = st.checkbox("使用默认数据文件 (young-diabetes.csv)", key="gini_default")
        
        if uploaded_file is not None or use_default:
            # 加载数据
            if use_default:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data("young-diabetes.csv")
            else:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data(uploaded_file=uploaded_file)
            
            if data is not None:
                # 准备数据
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                
                st.subheader("📊 数据概览")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("原始数据预览:")
                    try:
                        st.dataframe(raw_data.head())
                    except ImportError:
                        # 如果pyarrow有问题，使用表格显示
                        st.table(raw_data.head())
                
                with col2:
                    st.write(f"数据形状: {data.shape}")
                    st.write(f"特征数量: {len(feature_names)}")
                    st.write(f"目标变量: {target_name}")
                
                # 计算基尼系数
                st.subheader("🧮 基尼系数计算")
                
                if st.button("📈 计算基尼系数", type="primary"):
                    # 计算原始基尼系数
                    original_gini = calculate_gini(y)
                    
                    # 计算各特征的基尼系数
                    gini_results = []
                    for i, feature in enumerate(feature_names):
                        feature_gini = calculate_feature_gini(X, y, i)
                        information_gain = original_gini - feature_gini
                        
                        gini_results.append({
                            '特征名称': feature,
                            '基尼系数': feature_gini,
                            '信息增益': information_gain
                        })
                    
                    # 按信息增益排序
                    gini_results.sort(key=lambda x: x['信息增益'], reverse=True)
                    
                    # 显示结果
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📋 基尼系数表")
                        results_df = pd.DataFrame(gini_results)
                        try:
                            st.dataframe(results_df.style.format({
                                '基尼系数': '{:.4f}',
                                '信息增益': '{:.4f}'
                            }))
                        except ImportError:
                            # 如果pyarrow有问题，使用表格显示
                            st.table(results_df.round(4))
                        
                        st.metric("原始基尼系数", f"{original_gini:.4f}")
                    
                    with col2:
                        st.subheader("📊 特征重要性可视化")
                        
                        # 基尼系数柱状图
                        fig1 = px.bar(results_df, 
                                     x='基尼系数', 
                                     y='特征名称',
                                     orientation='h',
                                     title="各特征基尼系数",
                                     color='基尼系数',
                                     color_continuous_scale='RdYlBu')
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # 信息增益柱状图
                        fig2 = px.bar(results_df, 
                                     x='信息增益', 
                                     y='特征名称',
                                     orientation='h',
                                     title="各特征信息增益",
                                     color='信息增益',
                                     color_continuous_scale='RdYlGn')
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 基尼系数计算说明
                    st.subheader("📖 基尼系数说明")
                    st.info("""
                    **基尼系数 (Gini Impurity)**：衡量数据集不纯度的指标
                    - 值范围：0到0.5（二分类）
                    - 0表示完全纯净（所有样本属于同一类）
                    - 0.5表示最不纯净（各类样本数量相等）
                    
                    **信息增益 (Information Gain)**：使用某特征分割后基尼系数的减少量
                    - 值越大，该特征对分类的贡献越大
                    - 决策树算法优先选择信息增益大的特征进行分割
                    """)

    elif st.session_state.page_type == "performance_analysis":
        st.title("📈 模型性能分析")
        st.markdown("分析不同树深度下的模型性能")
        
        # 文件上传
        uploaded_file = st.file_uploader("📁 上传CSV文件", type=['csv'], key="perf_file")
        use_default = st.checkbox("使用默认数据文件 (young-diabetes.csv)", key="perf_default")
        
        if uploaded_file is not None or use_default:
            # 加载数据
            if use_default:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data("young-diabetes.csv")
            else:
                data, encoders, feature_names, target_name, original_values, raw_data = load_and_encode_data(uploaded_file=uploaded_file)
            
            if data is not None:
                st.subheader("⚙️ 参数设置")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_depth_range = st.slider("最大深度范围", 1, 20, (1, 10))
                with col2:
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
                with col3:
                    random_state = st.number_input("随机种子", value=42)
                
                if st.button("🚀 开始分析", type="primary"):
                    # 准备数据
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    
                    # 分割数据
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # 存储结果
                    results = []
                    trees = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    depths = range(max_depth_range[0], max_depth_range[1] + 1)
                    
                    for i, depth in enumerate(depths):
                        status_text.text(f'训练深度为 {depth} 的决策树...')
                        
                        # 训练模型
                        model = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
                        model.fit(X_train, y_train)
                        
                        # 预测
                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)
                        
                        # 计算准确率
                        train_acc = accuracy_score(y_train, train_pred)
                        test_acc = accuracy_score(y_test, test_pred)
                        
                        results.append({
                            '深度': depth,
                            '训练集准确率': train_acc,
                            '验证集准确率': test_acc,
                            '过拟合程度': train_acc - test_acc
                        })
                        
                        # 存储决策树用于可视化
                        trees[depth] = model
                        
                        # 更新进度条
                        progress_bar.progress((i + 1) / len(depths))
                    
                    status_text.text('分析完成！')
                    progress_bar.empty()
                    status_text.empty()
                    
                    # 显示结果
                    results_df = pd.DataFrame(results)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    # with col1:
                    #     st.subheader("📊 性能曲线")
                        
                    #     # 准确率曲线
                    #     fig = go.Figure()
                    #     fig.add_trace(go.Scatter(x=results_df['深度'], y=results_df['训练集准确率'],
                    #                            mode='lines+markers', name='训练集准确率',
                    #                            line=dict(color='blue', width=2)))
                    #     fig.add_trace(go.Scatter(x=results_df['深度'], y=results_df['验证集准确率'],
                    #                            mode='lines+markers', name='验证集准确率',
                    #                            line=dict(color='red', width=2)))
                        
                    #     fig.update_layout(title='准确率 vs 树深度',
                    #                     xaxis_title='树深度',
                    #                     yaxis_title='准确率',
                    #                     hovermode='x unified')
                    #     st.plotly_chart(fig, use_container_width=True)
                        
                    #     # 过拟合程度
                    #     fig2 = px.line(results_df, x='深度', y='过拟合程度',
                    #                   title='过拟合程度 vs 树深度',
                    #                   markers=True)
                    #     fig2.add_hline(y=0, line_dash="dash", line_color="red")
                    #     st.plotly_chart(fig2, use_container_width=True)
                    
                    # with col2:
                    #     st.subheader("📋 详细数据")
                    #     try:
                    #         st.dataframe(results_df.style.format({
                    #             '训练集准确率': '{:.3f}',
                    #             '验证集准确率': '{:.3f}',
                    #             '过拟合程度': '{:.3f}'
                    #         }))
                    #     except ImportError:
                    #         # 如果pyarrow有问题，使用表格显示
                    #         st.table(results_df.round(3))
                        
                    #     # 使用expander来隐藏/显示最佳结果，不会刷新页面
                    #     with st.expander("🎯 点击查看最佳结果", expanded=False):
                    #         # 最佳深度推荐
                    #         best_depth_idx = results_df['验证集准确率'].idxmax()
                    #         best_depth = results_df.loc[best_depth_idx, '深度']
                    #         best_acc = results_df.loc[best_depth_idx, '验证集准确率']
                            
                    #         st.success(f"🎯 推荐深度: {best_depth}")
                    #         st.success(f"📈 最佳验证准确率: {best_acc:.3f}")
                            

                    
                    # 决策树可视化
                    st.subheader("🌳 决策树可视化")
                    
                    # 使用标签页来展示不同深度的决策树
                    depth_tabs = st.tabs([f"深度 {depth}" for depth in trees.keys()])
                    
                    for i, depth in enumerate(trees.keys()):
                        with depth_tabs[i]:
                            st.write(f"**决策树深度: {depth}**")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # 绘制决策树
                                fig, ax = plt.subplots(figsize=(15, 10))
                                plot_tree(trees[depth], 
                                         feature_names=feature_names,
                                         class_names=encoders[target_name].classes_,
                                         filled=True, 
                                         rounded=True,
                                         fontsize=8)
                                plt.title(f'决策树 (深度={depth})')
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                # 显示该深度下的性能指标
                                depth_result = results_df[results_df['深度'] == depth].iloc[0]
                                st.metric("训练集准确率", f"{depth_result['训练集准确率']:.3f}")
                                st.metric("验证集准确率", f"{depth_result['验证集准确率']:.3f}")
                                st.metric("过拟合程度", f"{depth_result['过拟合程度']:.3f}")
                                
                                # 特征重要性
                                feature_importance = trees[depth].feature_importances_
                                importance_df = pd.DataFrame({
                                    '特征': feature_names,
                                    '重要性': feature_importance
                                }).sort_values('重要性', ascending=True)
                                
                                if len(importance_df) > 0:
                                    fig_imp = px.bar(importance_df, 
                                                   x='重要性', 
                                                   y='特征',
                                                   orientation='h',
                                                   title=f"特征重要性 (深度={depth})",
                                                   color='重要性',
                                                   color_continuous_scale='viridis')
                                    fig_imp.update_layout(height=300)
                                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # # 模型比较总结
                    # st.subheader("📝 模型分析总结")
                    
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                    #     # 使用expander来隐藏/显示关键发现
                    #     with st.expander("🔍 点击查看关键发现", expanded=False):
                    #         st.write("**关键发现:**")
                            
                    #         # 找出最佳模型
                    #         best_model = results_df.loc[results_df['验证集准确率'].idxmax()]
                    #         st.write(f"• 最佳验证准确率: {best_model['验证集准确率']:.3f} (深度={best_model['深度']})")
                            
                    #         # 找出过拟合最少的模型
                    #         min_overfit = results_df.loc[results_df['过拟合程度'].idxmin()]
                    #         st.write(f"• 最少过拟合: {min_overfit['过拟合程度']:.3f} (深度={min_overfit['深度']})")
                            
                    #         # 找出训练准确率最高的模型
                    #         max_train = results_df.loc[results_df['训练集准确率'].idxmax()]
                    #         st.write(f"• 最高训练准确率: {max_train['训练集准确率']:.3f} (深度={max_train['深度']})")
                    
                    # with col2:
                    #     st.write("**建议:**")
                        
                    #     if best_model['过拟合程度'] > 0.1:
                    #         st.warning("⚠️ 推荐模型可能存在过拟合，考虑使用更浅的树")
                    #     elif best_model['过拟合程度'] < -0.05:
                    #         st.info("ℹ️ 模型可能欠拟合，可以考虑增加树深度")
                    #     else:
                    #         st.success("✅ 推荐模型具有良好的泛化能力")
                        
                    #     # 稳定性分析
                    #     validation_std = results_df['验证集准确率'].std()
                    #     if validation_std > 0.05:
                    #         st.warning("⚠️ 验证准确率波动较大，模型稳定性有待提升")
                    #     else:
                    #         st.success("✅ 模型在不同深度下表现稳定")

# 页面底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    🌳 糖尿病预测决策树系统 | 基于机器学习的智能预测分析
    </div>
    """, 
    unsafe_allow_html=True
)
