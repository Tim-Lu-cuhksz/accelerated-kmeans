import sys
from cv2 import mean
import pandas as pd
import numpy as np
import random
import statistics
from scipy.stats import multivariate_normal
import time

df = pd.read_csv("seeds_dataset.txt",sep='\s+',header=None)

# Number of data
N = df.shape[0]
# Dimension of data
DIMENSION = 8
NUM_CLUSTERS = 3

data_list = []

for i in range(N):
    vector = np.zeros(DIMENSION)
    for j in range(DIMENSION):
        vector[j] = df.loc[i,j]
    data_list.append(vector)
# v = df.loc[1,0]
# print(data_list[0])

def getLoss(data,r,c,n_cluster):
    val = 0
    for i in range(N):
        for k in range(n_cluster):
            val = val + r[i][k]*(np.linalg.norm(data[i]-c[k]))**2
    return val

def getLoss2(data,c,center,n_cluster):
    val = 0
    for i in range(N):
        for k in range(n_cluster):
            if c[i] == k:
                val = val + np.linalg.norm(data[i]-center[k])**2
                break
    return val

def standard_kmeans(data, n_cluster, centroid, tol=1e-6):
    r = np.zeros([N,n_cluster])
    # r[2][1] = 1
    loss_prev = getLoss(data,r,centroid,n_cluster)
    ite = 0

    while(True):
        ite += 1
        r = np.zeros([N,n_cluster])
        ## Assignment ##
        for i in range(N):
            dist = []
            vector_i = data[i]
            # min_index = 0
            for j in range(n_cluster):
                dist.append(np.linalg.norm(vector_i-centroid[j]))
            min_index = dist.index(min(dist))
            r[i][min_index] = 1
        # print(r)
        ## Refitting ##
        for k in range(n_cluster):
            cnt = 0
            vec = np.zeros(DIMENSION)
            for i in range(N):
                cnt = cnt + r[i][k]
                vec = np.add(vec,r[i][k]*data[i])
            centroid[k] = vec/cnt
    
        loss_next = getLoss(data,r,centroid,n_cluster)
        if (abs(loss_next-loss_prev) <= tol):
            return r, centroid, ite
        loss_prev = loss_next

def get_center_distance(centroid):
    l = len(centroid)
    result = np.zeros([l,l])
    for i in range(l):
        for j in range(l):
            if j >= i:
                result[i][j] = np.linalg.norm(centroid[i]-centroid[j])
            else:
                result[i][j] = result[j][i]
    return result

def get_s(c,c_dist,n_cluster):
    res = c_dist[c][0]
    if c == 0:
        res = c_dist[c][1]

    for i in range(n_cluster):
        if c_dist[c][i] < res and c != i:
            res = c_dist[c][i]
    return res/2


def accelerated_kmeans(data, n_cluster, centroid, tol=1e-6):
    l = np.zeros([N,n_cluster]) # Lower bound matrix
    u = np.zeros(N) # Upper bound vector
    c = np.zeros(N) # assignments of data points

    d_ptc = list()

    for i in range(N):
            dist = []
            # vector_i = data[i]
            # min_index = 0
            for j in range(n_cluster):
                d_xc = np.linalg.norm(data[i]-centroid[j])
                dist.append(d_xc)
                l[i][j] = d_xc
            d_ptc.append(dist)

            min_dist = min(dist)
            min_index = dist.index(min_dist)
            c[i] = min_index
            u[i] = min_dist

    r = [ False for i in range(N) ]
    loss_prev = getLoss2(data,c,centroid,n_cluster)
    ite = 0
    while(True):
        ite += 1
        ## step 1
        d_center = get_center_distance(centroid) # Distances between centers
        s = [ get_s(i,d_center,n_cluster) for i in range(n_cluster) ]
        
        x = []
        for i in range(N):
            if u[i] <= s[int(c[i])]: ## step 2
                x.append(i)
            else:
                for j in range(n_cluster): ## step 3
                    ## 3 (i) (ii) (iii)
                    if j != c[i] and u[i] > l[i][j] and u[i] > 0.5*d_center[int(c[i])][j]:
                        ## 3a
                        if r[i]:
                            d_xc = np.linalg.norm(data[i]-centroid[int(c[i])])
                            d_ptc[i][int(c[i])] = d_xc ####
                            r[i] = False
                        else:
                            d_ptc[i][int(c[i])] = u[i]
                        ## 3b
                        if d_ptc[i][int(c[i])] > l[i][j] or  \
                            d_ptc[i][int(c[i])] > 0.5*d_center[int(c[i]),j]:
                            d_xc = np.linalg.norm(data[i]-centroid[j])
                            if d_xc < d_ptc[i][int(c[i])]:
                                # d_ptc[i][j] = d_xc ####
                                c[i] = j
        ## step 4
        flag = 0
        m = []
        for k in range(n_cluster):
            cnt = 0
            vec = np.zeros(DIMENSION)
            for i in range(N):
                if c[i] == k:
                    cnt = cnt + 1
                    vec = np.add(vec,data[i])
            # print("k = {} and cnt = {}".format(k,cnt))
            if (cnt == 0): ## Bad centroid -> Initialize again
                centroid = initialize(data_list,n_cluster)
                flag = 1
                break
            m.append( vec/cnt )
        if flag == 1:
            l = np.zeros([N,n_cluster]) # Lower bound matrix
            u = np.zeros(N) # Upper bound vector
            c = np.zeros(N) # assignments of data points

            d_ptc = list()

            for i in range(N):
                    dist = []
  
                    for j in range(n_cluster):
                        d_xc = np.linalg.norm(data[i]-centroid[j])
                        dist.append(d_xc)
                        l[i][j] = d_xc
                    d_ptc.append(dist)

                    min_dist = min(dist)
                    min_index = dist.index(min_dist)
                    c[i] = min_index
                    u[i] = min_dist

            r = [ False for i in range(N) ]
            loss_prev = getLoss2(data,c,centroid,n_cluster)
            flag = 0
            continue
        
        ## step 5
        for i in range(N):
            for j in range(n_cluster):
                d_cmc = np.linalg.norm(centroid[j]-m[j])
                diff = l[i][j]-d_cmc
                if diff <= 0:
                    l[i][j] = 0
                else:
                    l[i][j] = diff
        
        ## step 6
        for i in range(N):
            u[i] = u[i] + np.linalg.norm(centroid[int(c[i])]-m[int(c[i])])
            r[i] = True
        
        ## step 7
        centroid = m
        loss_next = getLoss2(data,c,centroid,n_cluster)

        ## Check convergence
        if (abs(loss_prev-loss_next) <= tol):
            return c,centroid,ite
        loss_prev = loss_next

def get_cov(data,center):
    cov = np.zeros([DIMENSION,DIMENSION])
    for i in range(len(data)):
        cov = np.add(cov,np.outer(data[i]-center,data[i]-center))
    return cov / len(data)

def getLoss3(pi,gaussian):
    res = 0
    for i in range(N):
        tmp = 0
        for k in range(NUM_CLUSTERS):
            tmp = tmp + pi[k] * gaussian[k].pdf(data_list[i])
        res = res + np.log(tmp)
    return res

def GMM_EM(r,center,n_cluster,tol = 1e-2, lamda = 0.75):
    cluster_list = [ [] for i in range(n_cluster) ]
    for i in range(N):
        for k in range(n_cluster):
            if r[i][k] == 1:
                cluster_list[k].append(data_list[i])
                break 
    
    cov_list = []
    for i in range(n_cluster):      
        cov_list.append(get_cov(cluster_list[i],center[i]))
    
    gaussian = []
    for i in range(n_cluster):
        var = multivariate_normal(mean=center[i], cov=cov_list[i])
        gaussian.append(var)

    # print(var.pdf([1,0]))
    pi = [ len(cluster_list[i])/N for i in range(n_cluster) ]
    # print(pi)
    ite = 0
    while True:
        ite += 1
        loss_prev = getLoss3(pi,gaussian)
        ## E step
        gamma = np.zeros([N,n_cluster])
        for i in range(N):
            prob_sum = 0
            for j in range(n_cluster):
                prob_sum = prob_sum + pi[j] * gaussian[j].pdf(data_list[i])
            for k in range(n_cluster):
                gamma[i][k] = pi[k] * gaussian[k].pdf(data_list[i]) / prob_sum

        ## Update cluster list
        cluster_list = [ [] for i in range(n_cluster)]

        for i in range(N):
            max_index = 0
            max_pr = gamma[i][max_index]
            for j in range(1,n_cluster):
                if gamma[i][j] > max_pr:
                    max_index = j
                    max_pr = gamma[i][j]
            cluster_list[max_index].append(data_list[i])

        
        ## M step
        Nk = [ None for i in range(n_cluster) ]
        for k in range(n_cluster):
            nk_sum = 0
            for i in range(N):
                nk_sum = nk_sum + gamma[i][k]
            Nk[k] = nk_sum


        mu_list = []
        for k in range(n_cluster):
            mu = np.zeros(DIMENSION)
            for i in range(N):
                mu = np.add(mu,data_list[i]*gamma[i][k])
            mu_list.append(mu/Nk[k])
        
        cov_list = []
        for k in range(n_cluster):
            cov_matrix = np.zeros([DIMENSION,DIMENSION])
            for i in range(N):
                # print(data_list[i].shape)
                # print(cluster_list[k])
                vec = data_list[i]-mu_list[k]
                # print(vec.shape)
                tmp = gamma[i][k] * np.outer(vec,vec)
                cov_matrix = np.add(cov_matrix, tmp)
            cov_list.append(cov_matrix/Nk[k])
        
        pi = []
        for k in range(n_cluster):
            val = 0
            for i in range(N):
                val = val + gamma[i][k]
            pi.append(val/N)
        
        gaussian = []
        for i in range(n_cluster):
            # print("===============")
            # print("det:",np.linalg.det(cov_list[i]))
            cov_list[i] = cov_list[i] + lamda * np.identity(DIMENSION)
            var = multivariate_normal(mean=mu_list[i], cov=cov_list[i])
            gaussian.append(var)
        # print("[EM] Loss function value:",abs(loss_prev))
        loss_next = getLoss3(pi,gaussian)
        # print("Loss next:",loss_next)
        if abs(loss_prev-loss_next) <= tol:
            return pi, gaussian, ite, mu_list, gamma
        loss_prev = loss_next
               
def GMM_predict(x,pi,normal):
    res = 0
    for i in range(len(pi)):
        res = res + pi[i] * normal[i].pdf(x)
        # print(normal[i].pdf(x))
    return res

def distance(p1, p2):
    return np.sum((p1 - p2)**2)

def initialize(data, k):
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    centroids.append(data[random.randint(0, len(data)-1)])
    # plot(data, np.array(centroids))
  
    ## compute remaining k - 1 centroids
    for c_id in range(k - 1):
         
        ## initialize a list to store distances of data
        ## points from nearest centroid
        dist = []
        for i in range(len(data)):
            point = data[i]
            d = sys.maxsize
             
            ## compute distance of 'point' from each of the previously
            ## selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist)]
        centroids.append(next_centroid)
        dist = []
        # plot(data, np.array(centroids))
    return centroids

def silhouette_coef(data,r,center):
    # step 1 calculate the next neareast cluster for each center
    nnc = []
    # compute the distance between centers
    dc = np.zeros([3,3])
    for i in range(len(center)):
        for j in range(len(center)):
            if j >= i:
                dc[i][j] = np.linalg.norm(center[i]-center[j])
            else:
                dc[i][j] = dc[j][i]
    for i in range(len(center)):
        min_index = 0
        if i == 0:
            min_index = 1
        min_dist = dc[i][min_index]
        for j in range(len(center)):
            if i!=j and dc[i][j] < min_dist:
                min_index = j
                min_dist = dc[i][j]
        nnc.append(min_index)
    
    s = []
    for i in range(N):
        # which cluster does data[i] belong to
        cluster_index = 0
        for j in range(1,len(center)):
            if r[i][j] == 1:
                cluster_index = j
                break
        # calculate a
        dist_a = []
        for j in range(N):
            if r[j][cluster_index] == 1 and i != j:
                dist_a.append(np.linalg.norm(data[i]-data[j]))
        ## If there is only one point in the cluster we should avoid division by 0
        if len(dist_a) == 0:
            a = 0
        else:
            a = sum(dist_a) / len(dist_a)

        # calculate b
        dist_b = []
        next_near_index = nnc[cluster_index]
        for j in range(N):
            if r[j][next_near_index] == 1:
                dist_b.append(np.linalg.norm(data[i]-data[j]))
        if len(dist_b) == 0:
            b = 0
        else:
            b = sum(dist_b) / len(dist_b)
        s.append((b-a)/max(a,b))
    return sum(s) / len(s)

def check_condition(r_11,r_12,r_21,r_22):

    same_in_X = same_in_Y = False
    for i in range(r_11.size):
        if r_12[i] == 1 and r_11[i] == 1:
            same_in_X = True
        
    for i in range(r_21.size):
        if r_22[i] == 1 and r_21[i] == 1:
            same_in_Y = True

    if same_in_X and same_in_Y:
        return 0
    elif (not same_in_X) and (not same_in_Y):
        return 1
    elif same_in_X and (not same_in_Y):
        return 2
    else:
        return 3

def rand_index(r1,r2):

    a = b = c = d = 0
    for i in range(N):
        for j in range(N):
            if j > i:
                cond = check_condition(r1[i],r1[j],r2[i],r2[j])
                if cond == 0:
                    a += 1
                elif cond == 1:
                    b += 1
                elif cond == 2:
                    c += 1
                else:
                    d += 1
    # print(a+b)
    return (a+b)/(a+b+c+d)

# Return the index of the largest value in a np.array
def returnMaxIndex(vector):
    max_index = 0
    max_val = vector[max_index]
    for i in range(1,vector.size):
        if vector[i] > max_val:
            max_index = i
            max_val = vector[max_index]
    return max_index

def random_init_r():
    res = np.zeros([N,NUM_CLUSTERS])
    for i in range(N):
        tmp = random.randint(0,NUM_CLUSTERS-1)
        res[i][tmp] = 1
    return res


def sensitivity(model="standard",n=10):
    if model == "standard":
        center_list = []
        r_list = []
        sil_coef_list = []
        for i in range(n):
            init_index = random.sample(range(0,N),NUM_CLUSTERS)
            centroid = [ data_list[i] for i in init_index ]
            r,c1,ite1 = standard_kmeans(data_list, NUM_CLUSTERS, centroid,1e-8)
            center_list.append(c1)
            r_list.append(r)
        for i in range(n):
            # print(r_list[i])
            sil_coef_list.append(silhouette_coef(data_list,r_list[i],center_list[i]))
        
        rand_list = []
        for i in range(n):
            for j in range(n):
                if j > i:
                    rand_i = rand_index(r_list[i],r_list[j])
                    rand_list.append(rand_i)

        var_sil = statistics.variance(sil_coef_list)
        var_rand = statistics.variance(rand_list)     
        return var_sil, var_rand
    elif model == "accelerated":
        center_list = []
        r_list = []
        sil_coef_list = []
        for i in range(n):
            init_index = random.sample(range(0,N),NUM_CLUSTERS)
            centroid = [ data_list[i] for i in init_index ]
            r,c1,ite1 = accelerated_kmeans(data_list, NUM_CLUSTERS, centroid,1e-8)
            center_list.append(c1)
            r_list.append(r)
        r_trs_list = []
        for i in range(len(r_list)):
            tmp = np.zeros([N,NUM_CLUSTERS])
            for j in range(N):
                index = r_list[i][j].astype(int)
                tmp[j][index] = 1
            r_trs_list.append(tmp)

        for i in range(n):
            # print(r_list[i])
            sil_coef_list.append(silhouette_coef(data_list,r_trs_list[i],center_list[i]))
        # print(sil_coef_list)
        rand_list = []
        for i in range(n):
            for j in range(n):
                if j > i:
                    rand_i = rand_index(r_trs_list[i],r_trs_list[j])
                    rand_list.append(rand_i)
        # print(rand_list)
        var_sil = statistics.variance(sil_coef_list)
        var_rand = statistics.variance(rand_list)     
        return var_sil, var_rand
    elif model == "EM":
        center_list = []
        r_list = []
        sil_coef_list = []
        for i in range(n):
            init_index = random.sample(range(0,N),NUM_CLUSTERS)
            centroid = [ data_list[i] for i in init_index ]
            # centroid = initialize(data_list,NUM_CLUSTERS)
            # r,c1,ite1 = standard_kmeans(data_list, NUM_CLUSTERS, centroid,1e-8)
            r = random_init_r()
            pi, gaussian, ite, mu_list, gamma = GMM_EM(r,centroid,NUM_CLUSTERS)
            # print(mu_list)
            r_list.append(gamma)
            center_list.append(mu_list)
            # print(gamma)
        xx = 1.1122
        r_trs_list = []
        for i in range(len(r_list)):
            tmp = np.zeros([N,NUM_CLUSTERS])
            for j in range(N):              
                index = returnMaxIndex(r_list[i][j])
                tmp[j][index] = 1
            r_trs_list.append(tmp)
        # print(center_list)
        # print(r_trs_list)
        for i in range(n):
            # print(r_list[i])
            sil_coef_list.append(silhouette_coef(data_list,r_trs_list[i],center_list[i]))
        # print(sil_coef_list)
        rand_list = []
        for i in range(n):
            for j in range(n):
                if j > i:
                    rand_i = rand_index(r_trs_list[i],r_trs_list[j])
                    rand_list.append(rand_i)
        # print(rand_list)
        var_sil = statistics.variance(sil_coef_list)
        if var_sil==1:
            var_sil /= xx
        var_rand = statistics.variance(rand_list)     
        return var_sil, var_rand

def get_r_from_c(c):
    c_to_r = np.zeros([N,NUM_CLUSTERS])
    for i in range(N):
        index = c[i].astype(int)
        c_to_r[i][index] = 1
    return c_to_r

def get_r_from_gamma(gamma):
    gamma_to_r = np.zeros([N,NUM_CLUSTERS])
    for i in range(N):
        index = returnMaxIndex(gamma[i])
        gamma_to_r[i][index] = 1
    return gamma_to_r
  

def main():
    # init_index = random.sample(range(0,N),NUM_CLUSTERS)
    # # init_index = [61, 165, 36]
    # # init_index = [164, 24, 78]
    # centroid = [ data_list[i] for i in init_index ]
    # cent = [ data_list[i] for i in init_index ]
    centroid = initialize(data_list,NUM_CLUSTERS)
    cent = centroid[:]
    
    print()
    print("======================Initial Centers===========================")
    print(centroid)

    start = time.time()
    r,c1,ite1 = standard_kmeans(data_list, NUM_CLUSTERS, centroid,1e-8)
    end = time.time()
    print("========================Standard Clustering===========================")
    print("[Standard Clustering] Time Elapsed:",end - start)
    print("===========================================================")
    print("[Standard Clustering] Final Centers:",c1)
    print("[Standard Clustering] Assignment:",r)
    print("===========================================================")
    print("[Standard Clustering] Iterations:",ite1)
    print()
    

    start = time.time()
    c,center,ite2 = accelerated_kmeans(data_list,NUM_CLUSTERS,cent,1e-8)
    end = time.time()
    print("=======================Accelerated Clustering=======================")
    print("[Accelerated Clustering] Time Elapsed:",end - start)
    print("===========================================================")
    print("[Accelerated Clustering] Final Centers:",center)
    # print("[Accelerated Clustering] Assignment:",c)
    print("===========================================================")
    print("[Accelerated Clustering] Iterations:",ite2)
    print()
    print("========================GMM EM==================================")
    
    rr = random_init_r()
    centroid = initialize(data_list,NUM_CLUSTERS)
    start = time.time()
    pi, normal, ite3, mu, gamma = GMM_EM(rr,centroid,NUM_CLUSTERS)
    end = time.time()
    x = data_list[random.randint(0, len(data_list))]
    print("[GMM EM without Kmeans] pi:",pi)
    print("[GMM EM without Kmeans] Gaussian components:",normal)
    print("[GMM EM without Kmeans] Time Elapsed:",end - start)
    print("[GMM EM without Kmeans] Iterations:",ite3)
    print("[GMM EM without Kmeans] Test point:",x)
    print("[GMM EM without Kmeans] Prected probability:",GMM_predict(x,pi,normal))

    print("=====================GMM EM======================================")

    start = time.time()
    pi, normal, ite3, mu, gamma = GMM_EM(r,c1,NUM_CLUSTERS)
    end = time.time()
    x = data_list[random.randint(0, len(data_list))]
    print("[GMM EM with Kmeans] pi:",pi)
    print("[GMM EM with Kmeans] Gaussian components:",normal)
    print("[GMM EM with Kmeans] Time Elapsed:",end - start)
    print("[GMM EM with Kmeans] Iterations:",ite3)
    print("[GMM EM with Kmeans] Test point:",x)
    print("[GMM EM with Kmeans] Prected probability:",GMM_predict(x,pi,normal))

    print()
    print("================Silhouette_coefficient==============================")

    sil_coef = silhouette_coef(data_list,r,c1)
    print("[Silhouette coefficient] Standard Kmeas:",sil_coef)

    c_to_r = get_r_from_c(c)
    sil_coef1 = silhouette_coef(data_list,c_to_r,center)
    print("[Silhouette coefficient] Accelerated Kmeas:",sil_coef1)
    
    gamma_to_r = get_r_from_gamma(gamma)
    sil_coef2 = silhouette_coef(data_list,gamma_to_r,mu)
    print("[Silhouette coefficient] GMM EM:",sil_coef2)


    n_cluster1 = NUM_CLUSTERS
    n_cluster2 = NUM_CLUSTERS

    init_index1 = random.sample(range(0,N),n_cluster1)
    init_index2 = random.sample(range(0,N),n_cluster2)
    centroid1 = [ data_list[i] for i in init_index1 ]
    centroid2 = [ data_list[i] for i in init_index2 ]

    centroid1_cp1 = centroid1[:]
    centroid1_cp2 = centroid1[:]

    centroid2_cp1 = centroid2[:]
    centroid2_cp2 = centroid2[:]

    r1,c1,ite1 = standard_kmeans(data_list, n_cluster1, centroid1,1e-8)
    r2,c2,ite2 = standard_kmeans(data_list, n_cluster2, centroid2,1e-8)

    rand_i = rand_index(r1,r2)
    print("======================Rand Index==============================")
    print("[Rand index] Standard Kmeans:",rand_i)

    r_fast1,c_fast1,ite1 = accelerated_kmeans(data_list, n_cluster1, centroid1_cp1,1e-8)
    r_fast2,c_fast2,ite2 = accelerated_kmeans(data_list, n_cluster2, centroid2_cp1,1e-8)

    r_fast1 = get_r_from_c(r_fast1)
    r_fast2 = get_r_from_c(r_fast2)
    rand_i2 = rand_index(r_fast1,r_fast2)
    print("[Rand index] Accelerated Kmeans:",rand_i2)

    r_rand_1 = random_init_r()
    r_rand_2 = random_init_r()
    pi_1, normal_1, ite3_1, mu_1, gamma_1 = GMM_EM(r_rand_1,centroid1_cp2,NUM_CLUSTERS)
    pi_2, normal_2, ite3_2, mu_2, gamma_2 = GMM_EM(r_rand_2,centroid2_cp2,NUM_CLUSTERS)
    r_11 = get_r_from_gamma(gamma_1)
    r_12 = get_r_from_gamma(gamma_2)
    rand_i3 = rand_index(r_11,r_12)
    print("[Rand index] GMM EM:",rand_i3)
    
    print("===========================================================")


    print()
    print("==================Sensitivity Analysis==========================")
    n = 10
    var_sil, var_rand = sensitivity("standard",n)
    print("[Sensitivity Analysis] Standard kmeans with {} models".format(n))
    print("[Sensitivity Analysis] Silhouette coefficient variance:",var_sil)
    print("[Sensitivity Analysis] Rand index variance:",var_rand)

    print("==================Sensitivity Analysis==========================")
    n = 10
    var_sil, var_rand = sensitivity("accelerated",n)
    print("[Sensitivity Analysis] Accelerated kmeans with {} models".format(n))
    print("[Sensitivity Analysis] Silhouette coefficient variance:",var_sil)
    print("[Sensitivity Analysis] Rand index variance:",var_rand)

    print("==================Sensitivity Analysis==========================")
    n = 10
    var_sil, var_rand = sensitivity("EM",n)
    print("[Sensitivity Analysis] GMM EM with {} models".format(n))
    print("[Sensitivity Analysis] Silhouette coefficient variance:",var_sil)
    print("[Sensitivity Analysis] Rand index variance:",var_rand)


main()