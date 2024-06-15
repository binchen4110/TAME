
import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict
import os


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################




def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, no_use = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, no_use = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score




def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

# 使用当前要预测的s掩码全局adj, 目的：相当于Attention, 指的是在学习过程中着重强调当前要预测的s
def mask_by_s(triplets, num_enity):
    # important!!!
    # 得到mask tensor, 后面掩码全局adj使用
    # 创建一个形状为 (n, 1) 的全零Tensor，n为节点数量
    mask_tensor = torch.zeros((num_enity, 1), dtype=torch.float32)
    src = triplets[:, 0].reshape(-1)
    src = torch.unique(src)

    # 将有边连接的节点在Tensor中对应的位置设置为1
    # 注意：这里使用了torch.unique来确保节点不会因为多条边而被重复设置
    mask_tensor[src.long(), 0] = 1
    
    return mask_tensor


def neighbors_2_hop_retrieve(g):
    node_neighbors_dict = {}
    # 先获取所有节点1跳邻居
    for node in g.nodes():
        # 获得从node出发的所有边的终点
        neighbors_1_hop = g.successors(node).tolist()
        # 获得所有到达node的边的起点
        neighbors_1_hop.extend(g.predecessors(node).tolist())
        neighbors_1_hop = list(set(neighbors_1_hop))
        node_neighbors_dict[node.item()] = {"1_hop": neighbors_1_hop}

    # 构建2跳图
    g_2 = dgl.khop_graph(g, k=2)

    # 再获取所有节点2跳邻居
    for node in g_2.nodes():
        # 获得从node出发的所有边的终点
        neighbors_2_hop = g_2.successors(node).tolist()
        # 获得所有到达node的边的起点
        neighbors_2_hop.extend(g_2.predecessors(node).tolist())
        neighbors_2_hop = list(set(neighbors_2_hop))

        node_neighbors_dict[node.item()].update({"2_hop": neighbors_2_hop})
    
    # 去重，防止又是1hop又是2hop的情况
    
    return node_neighbors_dict


# 构建dense graph, 实现2-hop邻居检索
def build_dense_graph(num_nodes, num_rels, input_list, curr_t, use_cuda, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm  = 1.0 / in_deg
        return norm
    
    triples = np.concatenate(input_list, axis=0) # 把各个时间戳的事件拼接起来
    # triples = triples[:, :3] # [s, r, o]
    triples = triples[:, :] # [s, r, o, t]
    src, rel, dst, t = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    t = np.abs(curr_t - t) #取时间间隔
    t = np.concatenate((t, t))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)

   
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edge: {'norm': edge.src['norm'] * edge.dst['norm']})
    g.edata['type']  = torch.LongTensor(rel)
    g.edata['ts']  = torch.FloatTensor(t)

    uniq_r, r_len, r_to_e = r2e(triples[:,:3], num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    # print(triples.shape)
    triples = triples[:, :3]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    
    # # important!!!
    # # 得到mask tensor, 后面掩码全局adj使用
    # # 创建一个形状为 (n, 1) 的全零Tensor，n为节点数量
    # mask_tensor = torch.zeros((num_nodes, 1), dtype=torch.float32)
    # # 将有边连接的节点在Tensor中对应的位置设置为1
    # # 注意：这里使用了torch.unique来确保节点不会因为多条边而被重复设置
    # mask_tensor[torch.unique(torch.from_numpy(src)).long(), 0] = 1
    # g.mask = mask_tensor


    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g

# 获取头部entity列表
def get_head_entity_list(my_data, head_ratio):
    sorted_indices = np.argsort(my_data[:,-1])[::-1] # 按频率实现降序排列
    sorted_data = my_data[sorted_indices]
    head_num = int(len(sorted_data)*head_ratio)
    head_data = sorted_data[:head_num,:]
    head_entity = head_data[:,0].reshape(-1)

    return list(head_entity)

# 取出不在head_list中的index
def split_array(base_array, refer_list):
    # base_array = base_array
    column_index = 2 # 要预测的o
    # mask = np.isin(base_array[:, column_index], refer_list)

    # head_index = np.where(mask)
    # tail_index = np.where(~mask)

    mask = torch.isin(base_array[:, column_index], refer_list)

    head_index = torch.where(mask)
    tail_index = torch.where(~mask)

    # array_in_list = base_array[mask]
    # array_not_in_list = base_array[~mask]

    return head_index, tail_index

# 根据某列的值把list分成两部分, 并返回相应bool标志
def split_array_for_ids(base_array, refer_list):
    column_index = 2 # 要预测的o
    mask = torch.isin(base_array[:, column_index], refer_list)

    # array_in_list = base_array[mask]
    # array_not_in_list = base_array[~mask]

    return np.array(mask), np.array(~mask)

# 从neighbors_dict中取出2-hop以内邻居
def get_neighbors(s_entity, neighbors_dict):

    neighbors_all = []
    s = s_entity.cpu().item()
    if s in neighbors_dict.keys():
        # 取出 1-hop, 2-hop邻居检索
        neighbors_1_hop = neighbors_dict[s]["1_hop"]
        neighbors_2_hop = neighbors_dict[s]["2_hop"]
        # 合并这两跳邻居（即，去重）
        neighbors_all = list(set(neighbors_1_hop + neighbors_2_hop))
    return neighbors_all

# 取1-hop邻居
def get_neighbors_1_hop(s_entity, neighbors_dict):

    neighbors_all = []
    s = s_entity.cpu().item()
    if s in neighbors_dict.keys():
        # 取出 1-hop邻居检索
        neighbors_1_hop = neighbors_dict[s]["1_hop"]
        # neighbors_2_hop = neighbors_dict[s]["2_hop"]
        # 合并这两跳邻居（即，去重）
        neighbors_all = list(set(neighbors_1_hop))
    return neighbors_all

# 取2-hop邻居    
def get_neighbors_2_hop(s_entity, neighbors_dict):

    neighbors_all = []
    s = s_entity.cpu().item()
    if s in neighbors_dict.keys():
        # 取2-hop邻居检索
        # neighbors_1_hop = neighbors_dict[s]["1_hop"]
        neighbors_2_hop = neighbors_dict[s]["2_hop"]
        # 合并这两跳邻居（即，去重）
        neighbors_all = list(set(neighbors_2_hop))
    return neighbors_all
    

# 针对每个snapshot，得到其s的邻居矩阵
def get_neighbors_matrix(triplets, neighbors, num_entity):
    s_entities = list(triplets[:, 0].reshape(-1))
    l = torch.stack(s_entities)
    l_values, _ = torch.sort(l)
    lst = [0 for _ in range(num_entity)]
    for ii in l_values:
        ii = ii.cpu().item()
        lst[ii] = 1
    
    neighbors_matrix = np.zeros((len(s_entities), num_entity))
    for i in range(len(s_entities)):
        s_entity = s_entities[i]
        # s = s_entity.cpu().item()
        # s_neighhbors_list = get_neighbors(s_entity, neighbors) # 2-hop以内
        s_neighhbors_list = get_neighbors_1_hop(s_entity, neighbors) # 1-hop以内
        neighbors_matrix[i, s_neighhbors_list] = 1
    
    return torch.Tensor(neighbors_matrix)
# 针对每个snapshot，得到其s的邻居矩阵
def get_neighbors_matrix_full(triplets, neighbors, num_entity):
    s_entities = list(triplets[:, 0].reshape(-1))
    l = torch.stack(s_entities)
    l_values, _ = torch.sort(l)
    lst = [0 for _ in range(num_entity)]
    for ii in l_values:
        ii = ii.cpu().item()
        lst[ii] = 1
    
    neighbors_matrix = np.zeros((num_entity, num_entity))
    for i in range(num_entity):
        tmp = lst[i]
        if tmp==1:
            s_entity = i
        # s = s_entity.cpu().item()
        # s_neighhbors_list = get_neighbors(s_entity, neighbors) # 2-hop以内
            s_neighhbors_list = get_neighbors_1_hop(s_entity, neighbors) # 1-hop以内
            neighbors_matrix[i, s_neighhbors_list] = 1
    
    return torch.Tensor(neighbors_matrix)
# 针对每个snapshot，得到其s的非邻居矩阵（即，有邻居点标记为0）
def get_neighbors_matrix_tranverse(triplets, neighbors, num_entity):
    s_entities = list(triplets[:, 0].reshape(-1))
    neighbors_matrix = np.ones((len(s_entities), num_entity))
    for i in range(len(s_entities)):
        s_entity = s_entities[i]
        # s = s_entity.cpu().item()
        # s_neighhbors_list = get_neighbors(s_entity, neighbors) # 2-hop以内
        s_neighhbors_list = get_neighbors_1_hop(s_entity, neighbors) # 1-hop以内
        neighbors_matrix[i, s_neighhbors_list] = 0
    
    return torch.Tensor(neighbors_matrix)


# 增加预测中tail实体的分数
def add_tail_score(triplets, scores, neighbors, head_entity_list):
    s_entity = list(triplets[:, 0].reshape(-1))
    # 1，2跳邻居score分别增大的倍数
    # 因为model.predict()得出的分数（取了log）是负数, 所以这里是除以

    multiple_factor = 1
    # multiple_factor_1_hop = 20
    # multiple_factor_2_hop = 5
    node_ids = list(range(scores.shape[1]))
    # 取tail实体
    tail_entity_list = [item for item in node_ids if item not in head_entity_list]
    tail_entity_set = set(tail_entity_list)
    for i in range(len(s_entity)):
        s = s_entity[i].cpu().item()
        if s in neighbors.keys():
            # 取出 1-hop, 2-hop邻居检索
            neighbors_1_hop = neighbors[s]["1_hop"]
            neighbors_2_hop = neighbors[s]["2_hop"]
            # 合并这两跳邻居（即，去重）
            neighbors_all = set(neighbors_1_hop + neighbors_2_hop)
            selected_o = neighbors_all & tail_entity_set
            selected_o_list = list(selected_o)
            scores[i, selected_o_list] /= multiple_factor


            # # 1-hop 
            # neighbors_1_hop = neighbors[s]["1_hop"]
            # # 又是尾实体又是1-hop邻居
            # selected_o_1_hop = set(neighbors_1_hop) & tail_entity_set
            # selected_o_1_hop_list = list(selected_o_1_hop)
            # scores[i, selected_o_1_hop_list] /=multiple_factor_1_hop

            # # 2-hop 
            # neighbors_2_hop = neighbors[s]["2_hop"]
            # # 又是尾实体又是2-hop邻居
            # selected_o_2_hop = set(neighbors_2_hop) & tail_entity_set
            # selected_o_2_hop_list = list(selected_o_2_hop)
            # scores[i, selected_o_2_hop_list] /=multiple_factor_2_hop
    
    return scores



# 1. 统计所有query中答案在其2-hop邻居内的比例；
# 2. 所有query,其2-hop邻居内head的占比；head query， tail query同理
def get_neighbor_and_head_ratio(test_triples, neighbors_dict, head_entity_list):

    no_neighbors_count = 0
    has_neighbors_count = 0
    is_anwser_in_neighbors = []
    is_anwser_in_neighbors_1_hop = []
    is_anwser_in_neighbors_2_hop = []
    head_in_neighbors_ratio_list = []
    has_answer_in_neighbors_id = []
    has_answer_in_neighbors_1_hop_id = []
    has_neighbors_id = []

    for i in range(len(test_triples)):
        # 统计所有query中答案在其2-hop邻居内的比例；
        s = test_triples[i, 0]
        o = test_triples[i, 2].cpu().item()
        neighbors_s = get_neighbors(s, neighbors_dict)
        neighbors_s_1_hop = get_neighbors_1_hop(s, neighbors_dict)
        neighbors_s_2_hop = get_neighbors_2_hop(s, neighbors_dict)

        if len(neighbors_s_1_hop) != 0: # s有1-hop邻居
            if o in neighbors_s_1_hop:
                is_anwser_in_neighbors_1_hop.append(1)
                has_answer_in_neighbors_1_hop_id.append(i)
            else:
                is_anwser_in_neighbors_1_hop.append(0)

        
        if len(neighbors_s_2_hop) != 0: # s有2-hop邻居
            if o in neighbors_s_2_hop:
                is_anwser_in_neighbors_2_hop.append(1)
            else:
                is_anwser_in_neighbors_2_hop.append(0)


        # if o in neighbors_s:
        #     is_anwser_in_neighbors.append(1)
        # else:
        #     is_anwser_in_neighbors.append(0)
        if len(neighbors_s) != 0: # s有邻居
            if o in neighbors_s:
                is_anwser_in_neighbors.append(1)
                has_answer_in_neighbors_id.append(i)
            else:
                is_anwser_in_neighbors.append(0)
            # 所有query,其2-hop邻居内head的占比
            head_entity_set = set(head_entity_list)
            is_neighbors_in_head = [(i in head_entity_set) for i in neighbors_s]
            neighbors_in_head_count = sum(is_neighbors_in_head)
            head_in_neighbors_ratio = neighbors_in_head_count / len(is_neighbors_in_head)
            head_in_neighbors_ratio_list.append(head_in_neighbors_ratio)
            has_neighbors_count +=1
            has_neighbors_id.append(i)
        else: # s无邻居
            head_in_neighbors_ratio_list.append(0)
            no_neighbors_count +=1

    return is_anwser_in_neighbors, is_anwser_in_neighbors_1_hop, is_anwser_in_neighbors_2_hop, head_in_neighbors_ratio_list, has_neighbors_count, no_neighbors_count, has_answer_in_neighbors_id, has_answer_in_neighbors_1_hop_id, has_neighbors_id

    

    






def get_total_rank_with_tail(test_triples, score, all_ans, neighbors_dict, head_entity_list, eval_bz, rel_predict=0, is_en=True):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if is_en:
            # 修改score_batch
            # triples_batch_np = triples_batch.cpu().nump()
            head_entity_list_ts = torch.tensor(head_entity_list).cuda()
            head_index, tail_index = split_array(triples_batch, head_entity_list_ts)
            triples_batch_tail = triples_batch[tail_index]
            score_batch_tail = score_batch[tail_index]
            added_score_tail = add_tail_score(triples_batch_tail, score_batch_tail, neighbors_dict, head_entity_list_ts)
            score_batch[tail_index] = added_score_tail


        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank

def get_total_rank(test_triples, score, all_ans, neighbors_dict, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    ave_rank = torch.mean(total_rank.float())
    print("-"*20)
    print("Ave ranks ({}): {:.6f}".format(method, ave_rank.item()))

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    hit_result = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_result.append(avg_count.item())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr.item(), hit_result, ave_rank


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        # print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap, nouse = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    times = set()
    for triple in data:
        times.add(triple[3])
    times = list(times)
    times.sort()
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list, np.asarray(times)


def slide_list(snapshots, k=1):
    k = k
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS23', 'ICEWS22', 'ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index, test_triples[_][3]])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0], test_triples[_][3]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a
def get_tail_neighbor(neighbors):
    a_num_dict = {}
    for key, value in neighbors.items():
        one_hop_list = value['1_hop']
        a_num = len(one_hop_list)
        a_num_dict[key] = a_num
    sorted_list = sorted(a_num_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_list = [x for x in sorted_list if x[1] != 0]
    total_num = len(sorted_list)
    keep_num = total_num // 2
    keep_list = [item[0] for item in sorted_list if item[1] <= 2]
    new_neighbors = {key: value for key, value in neighbors.items() if key in keep_list}
    return new_neighbors

def normalize_output(out_feat, idx):
    sum_m = 0
    a = torch.arange(0, idx, 1)
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[a], dim=1))
    return sum_m 
def link_dropout(adj, idx, k=5):
    adj = adj.numpy()
    type = adj.dtype
    adj = adj.astype(int)
    tail_adj = adj.copy()
    num_links = np.random.randint(k, size=idx) 
    num_links += 1

    for i in range(idx):
        index = tail_adj[i]
        new_idx = np.random.choice(index, num_links[i], replace=False)
        tail_adj[i] = 0.0
        for j in new_idx:
            tail_adj[i, j] = 1.0
    tail_adj = tail_adj.astype(type)
    tail_adj = torch.from_numpy(tail_adj)

    return tail_adj