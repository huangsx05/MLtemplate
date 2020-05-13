import pandas as pd
import numpy as np
import copy
from itertools import combinations # 排列组合

class iTree():
    # 类变量 - 树的模板
    tree_template_x = {}
    # 节点的信息：来自 - 现在 (是否需要分裂)- 去往 
    tree_template_x['from'] = {}
    tree_template_x['now'] = {}
    tree_template_x['whatif'] = {}
    tree_template_x['to'] = {}

    tree_template_x['from']['trace'] = [] # 空列表，用于存放树的路径

    tree_template_x['now']['samples'] = None # 当前数据集大小
    tree_template_x['now']['vars_available'] = []  # 当前可用的变量
    tree_template_x['now']['dummy_var_list'] = []  # 当前只有唯一值的变量
    tree_template_x['now']['current_layers'] = 0 # 当前的层
    tree_template_x['now']['is_leaf'] = 0  # 当前是否为叶子节点

    # 如果是分类树
    tree_template_x['now']['target_counts'] = None # 目标数
    tree_template_x['now']['non_target_counts'] = None  # 非目标数
    tree_template_x['now']['gini'] = None  # 基尼指数
    tree_template_x['now']['prob'] = None  # 概率值
    tree_template_x['now']['class'] = None  # 分类值
    # 如果是回归树
    tree_template_x['now']['mse'] = None  # 均方差

    tree_template_x['whatif']['compete_win_varname'] = None # 竞争获胜的变量
    tree_template_x['whatif']['compete_win_vartype'] = None  # 竞争获胜的变量类型 N, O, C
    tree_template_x['whatif']['compete_win_gini'] = None  # 分类树的指标
    tree_template_x['whatif']['compete_win_mse'] = None  # 回归树指标
    tree_template_x['whatif']['except_var_list'] = []  # （异常）需要排除掉的变量

    tree_template_x['to']['left_condition'] = None # 左子树分值的筛选条件
    tree_template_x['to']['right_condition'] = None  # 右子树分值的筛选条件

    tree_template_x['to']['left'] = None # 左子树
    tree_template_x['to']['right'] = None  # 右子树

    # 静态方法
    '''
    变量的离散化是一个比较独立的问题，变量可分为三类：
    - C(Continuous) 连续型 -> 百分位的有序离散 ：产生最多100种的二分切割
    - N(Nominal) 无序离散型 -> ∑C(n, x), x >= n//2 种二分切割
    - O(Ordinal) 有序离散型 -> 产生最多 N种的二分切割
    '''
    # 函数1:根据指定的起始和终止百分位，将一个Series进行切割
    @staticmethod
    def cbcut(data=None, pstart=0.1, pend=0.9):
        data = data.copy()
        # 搜索的分位数在10~90之间
        # 使用linspace计算每个分位点（0.1， 0.11，... 0.9)
        bins = int((pend - pstart) * 100 + 1)
        qlist = np.linspace(pstart, pend, bins)

        # 分位数可能有重复，去重（unique去重后会排序）
        qtiles = data.quantile(qlist).unique()
        res_list = []
        for q in qtiles:
            data1 = data.apply(lambda x: 1 if x < q else 0)
            res_list.append(data1)
        res_dict = {}
        res_dict['data_list'] = res_list
        res_dict['qtiles'] = qtiles
        return res_dict

    # 函数2：可以指定起始的相对位置 / 该函数也可以用于文本，但建议数据经过处理后是没有缺失的全数值状态
    @staticmethod
    def obcut(data=None, start=1):
        data = data.copy()
        qtiles = data.unique()[start:]
        res_list = []
        for q in qtiles:
            data1 = data.apply(lambda x: 1 if x < q else 0)
            res_list.append(data1)
        res_dict = {}
        res_dict['data_list'] = res_list
        res_dict['qtiles'] = qtiles
        return res_dict
    # 函数3：制作列表键值的映射字典 - 配合map使用
    @staticmethod
    def list_key_dict(data = None ,fill_value = 1):
        return dict(zip(data, [fill_value]*len(data)))
    # 函数4：字典键值互换（用作未来的反向映射）
    @staticmethod
    def kv2vk(data =None ):
        new_dict = {}
        for k in data.keys():
            new_dict[data[k]] = k 
        return new_dict
    # 函数5：离散无序型，就是排列组合
    @staticmethod
    def nbcut(data=None):
        data = data.copy()
        var_set = set(data)
        comb_num = len(var_set) // 2

        # 从var_set中生成指定数量的组合
        comb_list = []
        not_comb_list = []
        for i in range(comb_num):
            tem_num = i+1
            comb_list += list(combinations(var_set, tem_num))

        # 生成可用于选择组合的字典
        comb_sel_list = []
        for clist in comb_list:
            comb_sel_list.append(iTree.list_key_dict(data=clist))
            # 计算补集放到not_comb_list
            not_comb_list.append(list(var_set - set(clist)))

        res_list = []
        for comb_sel in comb_sel_list:
            data1 = data.map(comb_sel).fillna(0)
            res_list.append(data1)
        res_dict = {}
        res_dict['data_list'] = res_list  # 经过组合二分后的数据结果
        res_dict['comb_list'] = comb_list  # 原始的特征组合
        res_dict['not_comb_list'] = not_comb_list  # 特征组合的补集
        res_dict['comb_sel_list'] = comb_sel_list  # 根据特征组合给出的选择字典
        return res_dict
    # 函数6：数据属性收集
    @staticmethod
    def collect_var_attr(data=None, varname=None):
        data = data.copy()
        data1 = data.dropna().apply(str)
        # 1 缺失的数量
        missing_num = len(data) - data.notnull().sum()
        # 2 缺失率
        missing_rate = missing_num / len(data)

        # 3 唯一值
        levs = len(data1.unique())

        # 4 真-整型
        ## 4.1 全为数值为标准整型
        is_integer = data1.apply(lambda x: x.isdigit()).sum() == len(data1)

        # 5 是否全存在点
        is_dot = data1.apply(
            lambda x: True if '.' in x else False).sum() == len(data1)

        # 6 如果存在点，是否全为数值
        if is_dot:
            is_dot_is_digit = data1.apply(lambda x: True if x.split(
                '.')[0].isdigit() and x.split('.')[1].isdigit() else False).sum() == len(data1)
        else:
            is_dot_is_digit = False
        # 7 如果存在点，并且全为数值，第二部分是否等于0
        is_float = False
        if is_dot_is_digit:
            is_integer = data1.apply(lambda x: True if float(
                x.split('.')[1]) == 0 else False).sum() == len(data1)
        else:
            is_float = True
        # 尝试转换数值
        try:
            data1.apply(float)
            is_num = True
            is_str = False
        except:
            is_num = False
            is_str = True
        # 先赋值必然有的
        res_dict = {}
        res_dict['missing_num'] = missing_num
        res_dict['missing_rate'] = missing_rate

        res_dict['levs'] = levs
        res_dict['is_all_integer'] = is_integer
        res_dict['is_dot'] = is_dot
        res_dict['is_str'] = is_str
        res_dict['is_num'] = is_num

        res_dict['is_all_dot_and_digit'] = is_dot_is_digit
        res_dict['is_all_float'] = is_float
        return {varname: res_dict}
    # 函数7
    # Note:这里加一个变量的类型判定（C、N、O）后就可以执行后面的操作
    # 这里先写一个简单的规则判定
    @staticmethod
    def infer_var_type(data=None):
        if data['is_str']:
            res = 'N'
        else:
            if data['levs'] >=20:
                res = 'C'
            else:
                res = 'N'
        return res
    # 函数8：对df批量执行推断
    @staticmethod
    def df_infer_var_type(df, cols=None):
        var_meta_dict = {}
        if cols is None:
            cols = list(df.columns)
        for col in cols:
            tem_var_meta_dict = iTree.collect_var_attr(df[col], varname=col)
            var_meta_dict.update(tem_var_meta_dict)
        for k in var_meta_dict.keys():
            res = iTree.infer_var_type(var_meta_dict[k])
            var_meta_dict[k]['vartype'] = res
        return var_meta_dict

    # 函数9：将 x和y进行索引对齐(两个Series)，形成两列的df,丢掉空值
    # 注意，列名将改为x和y
    # x表示变量，y表示目标
    @staticmethod
    def align_xy(x = None, y = None ):
        tem_df = pd.DataFrame()
        tem_df['x'] = x.copy()
        tem_df['y'] = y.copy()
        return tem_df.dropna()
    
    # 函数10：计算gini不纯度
    # 我的Gini不纯度计算
    @staticmethod
    def cal_gini_impurity(target_count = None, total_count = None):
        p1 = target_count / total_count
        p0 = 1 - p1
        return 1 - p1**2 - p0**2
    
    # 函数11：计算gini
    @staticmethod
    def get_gini(x=None, y=None):
        tem_df = iTree.align_xy(x=x, y=y)
        # 计算分类值（按目前的预处理，分类值只会是两个）
        vals = tem_df['x'].unique()
        _gini = 0
        total_recs = len(tem_df)
        for val in vals:
            # 选出第一部分数据集
            tem_df1 = tem_df[tem_df['x'] == val]
            # 计算比重
            tem_weight = len(tem_df1) / total_recs
            # 目标数
            tem_target_count = tem_df1['y'].sum()
            # 该部分的gini
            tem_gini = iTree.cal_gini_impurity(
                target_count=tem_target_count, total_count=len(tem_df1))
            _gini += tem_weight * tem_gini
        return _gini
    # 函数12：计算mse
    # y_i 是真值
    # c_i 是该区域的均值
    @staticmethod
    def get_mse(x = None, y = None):
        tem_df = iTree.align_xy(x=x, y=y)
        # 计算分类值（按目前的预处理，分类值只会是两个）
        vals = tem_df['x'].unique()
        _mse = 0
        for val in vals:
            # 选出第一部分数据集
            tem_df1 = tem_df[tem_df['x'] == val]
            # 计算均值
            tem_df1_y_mean = tem_df1['y'].mean()
            # tem_mse = (tem_df1['y'] - tem_df1_y_mean).apply(lambda x: x**2).sum() / len(tem_df1['y']) # 应该是和而不是均
            tem_mse = (tem_df1['y'] - tem_df1_y_mean).apply(lambda x: x**2).sum() 
            _mse += tem_mse
        return _mse

    # 函数13：
    # 对于分类树，寻找最小gini
    # x是某个变量， y是目标列
    @staticmethod
    def find_min_gini(x=None, y=None, varname=None, vartype=None, start=1, pstart=0.1, pend=0.9):
        # 先获得可能的划分
        assert vartype in [
            'C', 'O', 'N'], 'Only Accept Vartype C(continuous), O(Oridinal), N(Nominal)'
        if vartype == 'N':
            res_dict = iTree.nbcut(data=x)
        elif vartype == 'O':
            res_dict = iTree.obcut(data=x, start=start)
        else:
            res_dict = iTree.cbcut(data=x, pstart=pstart, pend=pend)
        # 在循环上几种方式是一致的
        tem_gini_list = []
        for i in range(len(res_dict['data_list'])):
            tem_gini = iTree.get_gini(res_dict['data_list'][i], y)
            tem_gini_list.append(tem_gini)
        # index + min的方法得到最小值的位置
        min_gini = min(tem_gini_list)
        mpos = tem_gini_list.index(min_gini)
        if vartype == 'N':
            # 左边的部分（in)，仅仅是组合，筛选的时候还要套一次字典
            condition_left = res_dict['comb_list'][mpos]
            condition_right = res_dict['not_comb_list'][mpos]
        else:
            # 找到了某个值，将数据分为两份 （ < q 和 >= q)
            # 最终返回了最小的基尼指数，以及对应的划分条件
            condition_left = '<' + str(res_dict['qtiles'][mpos])
            condition_right = '>=' + str(res_dict['qtiles'][mpos])
        new_res_dict = {}
        new_res_dict[varname] = {}
        new_res_dict[varname]['gini'] = min_gini
        new_res_dict[varname]['condition_left'] = condition_left
        new_res_dict[varname]['condition_right'] = condition_right
        return new_res_dict

    # 函数14
    # 对于回归树，寻找最小mse
    # x是某个变量， y是目标列
    # https://blog.csdn.net/zpalyq110/article/details/79527653
    @staticmethod
    def find_min_mse(x=None, y=None, varname=None, vartype=None, start=1, pstart=0.1, pend=0.9):
        # 先获得可能的划分
        assert vartype in [
            'C', 'O', 'N'], 'Only Accept Vartype C(continuous), O(Oridinal), N(Nominal)'
        if vartype == 'N':
            res_dict = iTree.nbcut(data=x)
        elif vartype == 'O':
            res_dict = iTree.obcut(data=x, start=start)
        else:
            res_dict = iTree.cbcut(data=x, pstart=pstart, pend=pend)
        # 在循环上几种方式是一致的
        tem_mse_list = []
        for i in range(len(res_dict['data_list'])):
            tem_mse = iTree.get_mse(res_dict['data_list'][i], y)
            tem_mse_list.append(tem_mse)
        # index + min的方法得到最小值的位置
        min_mse = min(tem_mse_list)
        mpos = tem_mse_list.index(min_mse)
        if vartype == 'N':
            # 左边的部分（in)，仅仅是组合，筛选的时候还要套一次字典
            condition_left = res_dict['comb_list'][mpos]
            condition_right = res_dict['not_comb_list'][mpos]
        else:
            # 找到了某个值，将数据分为两份 （ < q 和 >= q)
            # 最终返回了最小的基尼指数，以及对应的划分条件
            condition_left = '<' + str(res_dict['qtiles'][mpos])
            condition_right = '>=' + str(res_dict['qtiles'][mpos])
        new_res_dict = {}
        new_res_dict[varname] = {}
        new_res_dict[varname]['mse'] = min_mse
        new_res_dict[varname]['condition_left'] = condition_left
        new_res_dict[varname]['condition_right'] = condition_right
        return new_res_dict
    
    # 函数15
    # 判断极值的函数
    @staticmethod
    def find_dict_minmax(some_dict = None, attrname = None, method='min'):
        klist = []
        vlist = []
        except_list = []
        for k in some_dict.keys():
            try:
                attr = some_dict[k][attrname]
                klist.append(k)
                vlist.append(attr)
            except:
                print('fail to find the val', k)
                except_list.append(k)
        # 极值
        if method.strip().lower() == 'min':
            the_val = min(vlist)
            the_key = klist[vlist.index(the_val)]
        elif method.strip().lower() == 'max':
            the_val = max(vlist)
            the_key = klist[vlist.index(the_val)]
        else:
            the_val, the_key = None, None
        return the_val, the_key, except_list

    # 函数16
    # 筛选和链式筛选
    # 筛选分为N和非N两种
    # N: 可以用集合筛选的办法，通过对选中的集合制作一个字典，然后map,保留非空的即可
    # 非N(O,C):采用判断符号的方法，进行数值计算筛选。要注意大小的判断都要符合左闭右开的原则。
    @staticmethod
    def filter_df_varattr(df=None, varname=None, vartype=None, condition=None):
        assert vartype in ['C', 'O', 'N'], 'Vartype must in C/O/N '
        if vartype == 'N':  # nomimal使用map挑选
            tem_map_dict = dict(zip(condition, [True]*len(condition)))
            _tem_sel = df[varname].map(tem_map_dict)
            res_df = df[_tem_sel.notnull()]
            # df['_tem_sel'] = df[varname].map(tem_map_dict)
            # res_df = df[df['_tem_sel'].notnull()]
        else:
            if '<' in condition:
                val = float(condition.replace('<', ''))
                res_df = df[df[varname] < val]
            elif '>=' in condition:
                val = float(condition.replace('>=', ''))
                res_df = df[df[varname] >= val]
            else:
                res_df = None
                raise ValueError('condition symbol error >=<', condition)
        # del df['_tem_sel']
        return res_df

    # 函数17
    # 链式筛选数据:由(varname, vartype, cut_condition)组成的字典列表
    @staticmethod
    def filter_df_varattr_chain(df=None, chain_list=None):
        tem_df = df.copy()
        for the_chain in chain_list:
            varname = the_chain['varname']
            vartype = the_chain['vartype']
            condition = the_chain['cut_condition']
            tem_df = iTree.filter_df_varattr(
                df=tem_df, varname=varname, vartype=vartype, condition=condition)
        return tem_df
    
    # 函数18
    # 随着数据集的不断拆分，有些变量可能只有唯一值，这样的变量不能再用于模型判断
    @staticmethod
    def find_dummy_var(df, cols=None, exe_cols=None):
        if cols is None:
            cols = list(df.columns)
        if exe_cols is not None:
            exe_cols1 = [x for x in exe_cols if len(x.strip()) > 0]
            cols = list(set(cols) - set(exe_cols1))
        res_list = []
        for c in cols:
            if len(df[c].unique()) == 1:
                res_list.append(c)
        return res_list




    # ============== 实例方法 =============
    def __init__(self):
        # 创建一个历史列表，记录历史记录 | 数据/参数/结果
        self.train_history_list = []
        self.train_branch_dict = None
        self.train_rules_df = None
        self.train_partition_df = None

        # debug
        self.debug = {}

    # 训练
    def fit(self, data=None, target_name=None, id_name=None, time_name=None, tree_type='classification',
            max_iter=1000, min_sample_to_split=100, min_sample_to_predict=10, max_depth=3, gini_thresh=0,
            mse_thresh=0, improve_ratio=0, class_thresh=0.5):
        assert all([not data.empty, target_name]), '必须输入数据集和变量名称'
        self.para_dict = {}
        self.para_dict['max_iter'] = max_iter  # 1 算法的最大循环次数
        # 2 当前数据集是否可分
        self.para_dict['min_sample_to_split'] = min_sample_to_split
        # 3 每个叶子节点的最小数量
        self.para_dict['min_sample_to_predict'] = min_sample_to_predict
        self.para_dict['max_depth'] = max_depth  # 4 树允许的最大深度
        self.para_dict['gini_thresh'] = gini_thresh  # 5 分类树的迭代停止阈值
        self.para_dict['mse_thresh'] = mse_thresh  # 6 回归树的迭代停止阈值
        self.para_dict['improve_ratio'] = improve_ratio  # 7 增益比/ 指标改善的百分比
        self.para_dict['class_thresh'] = class_thresh  # 8 分类的概率阈值
        self.para_dict['current_iter_list'] = []  # 9 算法运行需要的临时变量
        self.para_dict['history_dict'] = {}  # 10 保存整棵树的生长过程和数据
        # 11 叶子节点列表（决策规则） - 由于有些变量没有分割，因此部分需要回溯到倒数第二级几点
        self.para_dict['leaf_list'] = []
        self.para_dict['data'] = data  # 12 数据集，pd.DataFrame
        self.para_dict['target_name'] = target_name  # 13 目标变量名称
        self.para_dict['id_name'] = id_name  # 14 ID变量名称
        self.para_dict['time_name'] = time_name  # 15 时间变量名称
        self.para_dict['var_meta'] = iTree.df_infer_var_type(self.para_dict['data'])  # 16 计算变量的元信息
        self.para_dict['tree_type'] = tree_type  # 17 决策树类型

        # --- 以下执行算法 
        # 1 首次执行
        iter_cnt = 0
        tree_template, tem_df = self.first_head()
        tree_template = self.iter_body(tree_template = tree_template, tem_df = tem_df)
        self.para_dict['history_dict'] = tree_template
        while len(self.para_dict['current_iter_list']) > 0:
            iter_cnt += 1
            if (iter_cnt) >= self.para_dict['max_iter']:
                break
            tree_template, tem_df = self.iter_head()
            _ = self.iter_body(tree_template=tree_template, tem_df=tem_df)
        self.summary()

    # 循环时（非首次）数据的获取
    def first_head(self):
        tree_template = copy.deepcopy(self.tree_template_x)
        tem_df = self.para_dict['data']  # 直接读取 | 由于算法不会“破坏”数据，因此可以引用
        target_name = self.para_dict['target_name']
        exe_cols = [target_name, self.para_dict['id_name'], self.para_dict['time_name']]
        exe_cols = [x for x in exe_cols if x] # 去掉缺失变量
        tree_template['whatif']['except_var_list'] = list(set(tree_template['whatif']['except_var_list'] + exe_cols))
        tree_template['now']['samples'] = len(tem_df)
        tree_template['now']['dummy_var_list'] = iTree.find_dummy_var(tem_df, exe_cols= exe_cols)
        # 可用变量要去掉已定义好的特殊变量，傻变量，以及之前异常的变量（初始情况下为空）
        tree_template['now']['vars_available'] = list(set(tem_df.columns) - set(tree_template['now']['dummy_var_list']) -
                                                  set(tree_template['whatif']['except_var_list']))
        return tree_template, tem_df
    # 循环的数据获取
    def iter_head(self):
        tree_template = self.para_dict['current_iter_list'].pop()
        chain_list = tree_template['from']['trace']
        tem_df = iTree.filter_df_varattr_chain(df=self.para_dict['data'], chain_list=chain_list)
        tree_template['whatif']['except_var_list'] = list(set(tree_template['whatif']['except_var_list'] + [x['varname'] for x in chain_list]))
        tree_template['now']['samples'] = len(tem_df)
        tree_template['now']['dummy_var_list'] = iTree.find_dummy_var(tem_df)
        # 可用变量要去掉已定义好的特殊变量，傻变量，以及之前异常的变量（初始情况下为空）
        tree_template['now']['vars_available'] = list(set(tem_df.columns) -
                                                    set(tree_template['now']['dummy_var_list']) -
                                                    set(tree_template['whatif']['except_var_list']))
        return tree_template, tem_df


    # 循环体
    def iter_body(self, tree_template = None, tem_df = None):
        target_name = self.para_dict['target_name']
        # 要确保当前数据集的数量可分割，且有可分割的变量，并且层数未达上限
        is_enough_sample = tree_template['now']['samples'] >= self.para_dict['min_sample_to_split']
        is_depth_ok = tree_template['now']['current_layers'] < self.para_dict['max_depth']
        is_var_available = len(tree_template['now']['vars_available']) > 0
        # 计算当前的关键指标，看是否需要 “变量竞争”
        if self.para_dict['tree_type'] == 'classification':
            tree_template['now']['target_counts'] = tem_df[target_name].apply(int).sum()
            tree_template['now']['non_target_counts'] = tree_template['now']['samples'] - tree_template['now']['target_counts']
            tree_template['now']['gini'] = iTree.cal_gini_impurity(tree_template['now']['target_counts'], tree_template['now']['samples'])
            tree_template['now']['prob'] = tree_template['now']['target_counts'] / tree_template['now']['samples']
            tree_template['now']['class'] = 1 if tree_template['now']['prob'] >= self.para_dict['class_thresh'] else 0
            # 如果低于给定的阈值，那么就停止（事实上gini和mse都>=0)
            if tree_template['now']['gini'] < self.para_dict['gini_thresh']:
                is_kpi_need_improve = False
            else:
                is_kpi_need_improve = True
        else:
            y_mean = tem_df[target_name].mean()
            y = tem_df[target_name]
            tree_template['now']['mse'] = (y - y_mean).apply(lambda x: x**2).mean()
            if tree_template['now']['mse'] < self.para_dict['mse_thresh']:
                is_kpi_need_improve = False
            else:
                is_kpi_need_improve = True
        if all([is_enough_sample, is_depth_ok, is_var_available, is_kpi_need_improve]):
            is_compete = True
        else:
            is_compete = False
        # ====
        # 变量竞争 | 通过跑根节点的数据获得两个信息，1：有些信息是埋在变量里的Name -> Title 产生了更好的变量；2.有些变量处理后是可用的，例如将船舱分为有和无
        if is_compete:
            # 如果N型变量的值过多，不允许计算（会产生组合爆炸）
            N_too_many_lev = [x for x in tree_template['now']['vars_available'] if self.para_dict['var_meta'][x]['vartype'] == 'N' and len(tem_df[x].unique()) > 20]
            tree_template['whatif']['except_var_list'] = tree_template['whatif']['except_var_list'] + N_too_many_lev
            compete_dict = {}
            cols = list(
                set(tree_template['now']['vars_available']) - set(N_too_many_lev))
            for c in cols:
                tem_varname = c
                tem_vartype = self.para_dict['var_meta'][c]['vartype']
                print(tem_varname, tem_vartype)
                if self.para_dict['tree_type'] == 'classification':
                    tem_res_dict = iTree.find_min_gini(x=tem_df[c], y=tem_df[target_name], varname=tem_varname, vartype=tem_vartype)
                else:
                    tem_res_dict = iTree.find_min_mse(x=tem_df[c], y=tem_df[target_name], varname=tem_varname, vartype=tem_vartype)
                compete_dict.update(tem_res_dict)
            if self.para_dict['tree_type'] == 'classification':
                win_gini, win_var, except_list = iTree.find_dict_minmax(some_dict=compete_dict, attrname='gini')
                tree_template['whatif']['compete_win_varname'] = win_var
                tree_template['whatif']['compete_win_gini'] = win_gini
                tree_template['whatif']['except_var_list'] = tree_template['whatif']['except_var_list'] + except_list
                # 这里可以加入win_gini增益比的判断
            else:
                win_mse, win_var, except_list = iTree.find_dict_minmax(some_dict=compete_dict, attrname='mse')
                tree_template['whatif']['compete_win_varname'] = win_var
                tree_template['whatif']['compete_win_mse'] = win_mse
                tree_template['whatif']['except_var_list'] = tree_template['whatif']['except_var_list'] + except_list
            # 观察候选左右子树的情况
            # 决定是否创建左右子树 - 在非根节点的情况下要循环的去过滤（链式过滤），判断数据集个数
            # Note:如果不对变量处理，那么会产生冗余的分支（例如Embarked的左子树只有两条记录，是nan分出来的）
            win_dict = compete_dict[win_var]
            left_candidate_df = iTree.filter_df_varattr(df=tem_df, varname=win_var,
                                                vartype=self.para_dict['var_meta'][win_var]['vartype'],
                                                condition=win_dict['condition_left'])
            is_left_branch_ok = len(left_candidate_df) >= self.para_dict['min_sample_to_predict']
            right_candidate_df = iTree.filter_df_varattr(df=tem_df, varname=win_var,
                                                vartype=self.para_dict['var_meta'][win_var]['vartype'],
                                                condition=win_dict['condition_right'])
            is_right_branch_ok = len(right_candidate_df) >= self.para_dict['min_sample_to_predict']

            # 如果左右子树的数量都不足以作为一个叶子，那么就不分割
            # if not all([is_left_branch_ok, is_right_branch_ok]):
            #  -- 这里逻辑不对， 例如，左不可分，右可分 all[True, False] = False ,not之后变为True，结果枝节点被标为了叶子节点
            if not is_left_branch_ok and not is_right_branch_ok:
                tree_template['now']['is_leaf'] = 1
                self.para_dict['leaf_list'].append(tree_template)

            # 如果左子树可分
            if is_left_branch_ok:
                tem_left_dict = copy.deepcopy(self.tree_template_x)

                # 继承当前的trace
                left_trace = tree_template['from']['trace'].copy()
                tem_trace_dict = {}
                tem_trace_dict['varname'] = win_var
                tem_trace_dict['vartype'] = self.para_dict['var_meta'][win_var]['vartype']
                tem_trace_dict['cut_condition'] = win_dict['condition_left']
                left_trace.append(tem_trace_dict)

                # 更新模板中的trace
                tem_left_dict['from']['trace'] = left_trace
                tem_left_dict['now']['current_layers'] = tree_template['now']['current_layers'] + 1

                # 更新剔除的列表（父节点不能用的子节点也不能）
                tem_left_dict['whatif']['except_var_list'] = tree_template['whatif']['except_var_list']

                # 连接
                tree_template['to']['left_condition'] = win_dict['condition_left']
                tree_template['to']['left'] = tem_left_dict

                # 推入待执行列表
                self.para_dict['current_iter_list'].append(tree_template['to']['left'])

            # 如果右子树可分
            if is_right_branch_ok:
                tem_right_dict = copy.deepcopy(self.tree_template_x)

                # 继承当前的trace
                right_trace = tree_template['from']['trace'].copy()
                tem_trace_dict = {}
                tem_trace_dict['varname'] = win_var
                tem_trace_dict['vartype'] = self.para_dict['var_meta'][win_var]['vartype']
                tem_trace_dict['cut_condition'] = win_dict['condition_right']
                right_trace.append(tem_trace_dict)

                # 更新模板中的trace
                tem_right_dict['from']['trace'] = right_trace
                tem_right_dict['now']['current_layers'] = tree_template['now']['current_layers'] + 1

                # 更新剔除的列表（父节点不能用的子节点也不能）
                tem_right_dict['whatif']['except_var_list'] = tree_template['whatif']['except_var_list']

                # 连接
                tree_template['to']['right_condition'] = win_dict['condition_right']
                tree_template['to']['right'] = tem_right_dict

                # 推入待执行列表
                self.para_dict['current_iter_list'].append(tree_template['to']['right'])
        else:
            # 如果不需要竞争，则终止该节点为终止节点
            tree_template['now']['is_leaf'] = 1
            self.para_dict['leaf_list'].append(tree_template)
        return tree_template

    # 提取fit之后的结果
    def summary(self):
        self.train_branch_dict  ={}
        self.train_branch_dict['tier1'] = {} # 叶子节点
        self.train_branch_dict['tier2'] = {}  # 叶子的上一层节点
        leaf_list = self.para_dict['leaf_list']
        res_dict = {}
        res_dict1 = {}
        for i in range(len(leaf_list)):
            cx = 'b' + str(i)
            chain_list = leaf_list[i]['from']['trace']
            res_dict[cx] = {}
            res_dict[cx]['data'] = iTree.filter_df_varattr_chain(df=self.para_dict['data'], chain_list=chain_list)
            res_dict[cx]['trace'] = chain_list
            res_dict[cx]['proba'] = res_dict[cx]['data'][self.para_dict['target_name']].mean()
            res_dict[cx]['class'] = 1 if res_dict[cx]['proba'] >= self.para_dict['class_thresh'] else 0
            res_dict[cx]['size'] = len(res_dict[cx]['data'])
            # 如果链条长度大于1，说明有父节点
            if len(chain_list) > 1:
                cx1 = cx + '_a1'
                chain_list1  = chain_list[:-1]
                res_dict1[cx1] = {}
                res_dict1[cx1]['data'] = iTree.filter_df_varattr_chain(df=self.para_dict['data'], chain_list=chain_list1)
                res_dict1[cx1]['trace'] = chain_list1
                res_dict1[cx1]['proba'] = res_dict1[cx1]['data'][self.para_dict['target_name']].mean()
                res_dict1[cx1]['class'] = 1 if res_dict1[cx1]['proba'] >= self.para_dict['class_thresh'] else 0
                res_dict1[cx1]['size'] = len(res_dict1[cx1]['data'])

        self.train_branch_dict['tier1'] = res_dict
        self.train_branch_dict['tier2'] = res_dict1

        res_df = pd.DataFrame(columns = ['partition','size','proba','class','tier'])


        for k in res_dict.keys():
            tem_dict = {}
            tem_dict['partition'] = k
            tem_dict['size'] = res_dict[k]['size']
            tem_dict['proba'] = res_dict[k]['proba']
            tem_dict['class'] = res_dict[k]['class']
            tem_dict['tier'] = 'tier1'
            res_df = res_df.append(tem_dict, ignore_index=True)
        for k in res_dict1.keys():
            tem_dict = {}
            tem_dict['partition'] = k
            tem_dict['size'] = res_dict1[k]['size']
            tem_dict['proba'] = res_dict1[k]['proba']
            tem_dict['class'] = res_dict1[k]['class']
            tem_dict['tier'] = 'tier2'
            res_df = res_df.append(tem_dict, ignore_index=True)
        
        self.train_partition_df = res_df
        # 规则抽取
        rule_df = pd.DataFrame(columns=['rule', 'tier' ,'condition', 'class', 'proba','support'])

        for k in res_dict.keys():
            k_trace = res_dict[k]['trace']
            tem_dict = {}
            tem_dict['rule'] = k
            tem_dict['tier'] = 1
            tem_dict['condition'] = '&'.join([x['varname'] + ' in ' + str(list(x['cut_condition'])) if x['vartype'] == 'N' else x['varname'] +  x['cut_condition'] for x in k_trace ])
            tem_dict['class'] = res_dict[k]['class']
            tem_dict['proba'] = res_dict[k]['proba']
            tem_dict['support'] = res_dict[k]['size']
            rule_df = rule_df.append(tem_dict, ignore_index=True)

            if len(k_trace) > 1:
                # 上一级目录
                tem_dict = {}
                k_trace1 = k_trace[:-1]
                tem_dict['rule'] = k + '_a1'
                tem_dict['tier'] = 2
                tem_dict['condition'] = '&'.join([x['varname'] + ' in ' + str(list(x['cut_condition']))
                                            if x['vartype'] == 'N' else x['varname'] + x['cut_condition'] for x in k_trace1])
                tem_dict['proba'] = res_dict[k]['proba']
                tem_dict['class'] = res_dict[k]['class']
                tem_dict['support'] = res_dict[k]['size']
                rule_df = rule_df.append(tem_dict, ignore_index=True)
        self.train_rules_df = rule_df


    # 进行预测
    def predict(self, data=None):
        # 第一层预测（尽量用叶子节点）
        predict_df_list1 = [] # 叶子节点的预测结果
        predict_df_list2 = [] # 叶子的上一层节点的预测结果
        for b in self.train_branch_dict['tier1'].keys():
            # 获取训练得到的几个数据
            tem_proba = self.train_branch_dict['tier1'][b]['proba']
            tem_class = self.train_branch_dict['tier1'][b]['class']
            tem_size = self.train_branch_dict['tier1'][b]['size']
            tem_chain_list = self.train_branch_dict['tier1'][b]['trace']

            # 分区即预测
            tem_predict_df = iTree.filter_df_varattr_chain(df = data, chain_list= tem_chain_list)
            tem_predict_df['predict_class'] = tem_class
            tem_predict_df['predict_proba'] = tem_proba
            tem_predict_df['predict_support'] = tem_size

            # debug 
            tem_predict_df['branch'] = b

            predict_df_list1.append(tem_predict_df)
        # 这里有重复的index
        predict_df1 = pd.concat(predict_df_list1)
        
        # self.debug['a1'] = predict_df1
        # predict_df1 = predict_df1[~predict_df1.index.duplicated()]
        print('***totle recs to predict', len(data))
        print('***predict by leaf Node', predict_df1.shape)
        add_list = ['predict_proba', 'predict_class', 'predict_support']
        keep_list= [self.para_dict['target_name']] + add_list
        for x in add_list:
            data[x] = predict_df1[x]

        # 如果有数据未被叶子预测，则使用它的上一条
        is_missing_predict = data['predict_proba'].notnull().sum() < len(data)
        if is_missing_predict:
            mis_data = data[~data['predict_proba'].notnull()]
            for b in self.train_branch_dict['tier2'].keys():
                tem_proba = self.train_branch_dict['tier2'][b]['proba']
                tem_class = self.train_branch_dict['tier2'][b]['class']
                tem_size = self.train_branch_dict['tier2'][b]['size']
                tem_chain_list = self.train_branch_dict['tier2'][b]['trace']

                # 分区即预测
                tem_predict_df = iTree.filter_df_varattr_chain(df=mis_data, chain_list=tem_chain_list)
                tem_predict_df['predict_class'] = tem_class
                tem_predict_df['predict_proba'] = tem_proba
                tem_predict_df['predict_support'] = tem_size
                predict_df_list2.append(tem_predict_df)
            predict_df2 = pd.concat(predict_df_list2)
            # 这里的每条记录可能会重复预测?
            predict_df2 = predict_df2[~predict_df2.index.duplicated()]
            print('***predict by branch Node', predict_df2.shape)
            for x in add_list:
                mis_data[x] = predict_df2[x]
            data = pd.concat([data[data['predict_proba'].notnull()],mis_data])
        return data[keep_list].sort_index()
        # return data