import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from copy import deepcopy

from selection.selector import Selector

class SE_Kernel(torch.nn.Module):
    """
    Module to calculate the Squared Exponential Kernel.
    """
    def __init__(self,init_sigma = 1.0,l = 1.0,device=torch.device('cpu')):
        super(SE_Kernel, self).__init__()
        self.sigma_f = Parameter(torch.tensor(init_sigma))
        self.l = l
        self.device=device

    def forward(self,xs):
        """
        Return the Sigma(Covariance Matrix) with the given distance matrix r
        where r_{ij} = (x_i-x_j)'(x_i-x_j) 
        """
        x_size = xs.size(1)
        A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones([x_size,x_size],device=self.device)
        R = A+A.transpose(0,1)-2*(xs.transpose(0,1)).mm(xs)
        return torch.exp(-0.5*R/self.l)*self.sigma_f**2


class Poly_Kernel(torch.nn.Module):
    """
    Module to calculate the Polynomial Kernel
    """
    def __init__(self,order = 1,poly_norm = False,device=torch.device('cpu')):
        super(Poly_Kernel,self).__init__()
        self.order = order
        self.Normalize = poly_norm
        
        # Add an additional variance parameter
        self.sigma_f = Parameter(torch.tensor(1.0))

        self.device=device
        
    def forward(self,xs):
        """
        Return the covariance of x = [x1,x2]
        """
        k = (xs.transpose(0,1)).mm(xs)
        if self.Normalize:
            # Make K(x,x) = 1
            x_size = xs.size(1)
            A = torch.sum(xs**2,dim=0,keepdim=True)*torch.ones([x_size,x_size],device=self.device)
            k = k/torch.sqrt(A)/torch.sqrt(A.transpose(0,1))
            return torch.pow(k,self.order)*self.sigma_f**2
        else:
            return torch.pow(k,self.order)


class GPR(torch.nn.Module):
    """
    Gaussian Process Regression Model For Federated Learning
    This is a base class and a Covariance() function must be implemented by its subclasses

    Parameters:     
        Define by the kernel/covariance matrix
    
    Non-Parameter Tensor:
        noise: Standard Deviation of sample noise (sigma_n). 
               This noise is only to avoid singular covariance matrix.
        mu: Mean Priori, which is fixed while training. It can be evaluated by MLE with weighted data
    """
    def __init__(self,num_users,loss_type = 'MML',init_noise = 0.01,gamma=1.0,rho = None,device = torch.device('cpu')):
        """
        Arguments:
            num_users: Number of users in a Federated Learning setting
            loss_type: {"LOO","MML"}
        """
        super(GPR, self).__init__()
        self.num_users = num_users
        self.loss_type = loss_type
        # sigma_n
        self.noise = torch.tensor(init_noise,device=device)
        self.mu = torch.zeros(num_users,device=device).detach()
        self.discount = torch.ones(num_users,device=device).detach()
        self.data = {}
        self.gamma = gamma
        self.device = device

        # system efficiency stat
        self.consumption = torch.ones(num_users,device=device).detach()
        self.rho = rho

    def Covariance(self,ids = None):
        raise NotImplementedError("A GPR class must have a function to calculate covariance matrix")

    def Update_Training_Data(self,client_idxs,loss_changes,epoch):
        """
        The training data should be in the form of : data[epoch] = sample_num x [user_indices, loss_change] (N x 2)
        Thus the data[epoch] is in shape S x N x 2
        """
        data = np.concatenate([np.expand_dims(np.array(client_idxs),2),np.expand_dims(np.array(loss_changes),2)],2)
        self.data[epoch] = torch.tensor(data,device=self.device,dtype=torch.float)

    def Update_System_Stat(self,client_idxs,stat):
        if self.rho is not None:
            self.consumption[client_idxs]=torch.tensor(stat,device = self.device,dtype=torch.float)     

    def Reset_Discount(self):
        self.discount = torch.ones(self.num_users,device=self.device).detach()

    def Update_Discount(self,index,factor=0.9):
        self.discount[index]*=factor   

    def Posteriori(self,data):
        """
        Get the posteriori with the data
        data: given in the form [index,loss]
        return:mu|data,Sigma|data
        """
        data = torch.tensor(data,device=self.device,dtype=torch.float)
        indices = data[:,0].long()
        values = data[:,1]
        
        Cov = self.Covariance()
        
        Sigma_inv = torch.inverse(Cov[indices,:][:,indices])
        mu = self.mu+((Cov[:,indices].mm(Sigma_inv)).mm((values-self.mu[indices]).unsqueeze(1))).squeeze()
        Sigma = Cov-(Cov[:,indices].mm(Sigma_inv)).mm(Cov[indices,:])
        return mu.detach(),Sigma.detach()


    def Log_Marginal_Likelihood(self,data,weight=None):
        """
        MML:
        Calculate the log marginal likelihood of the given data
        data: given in the form S x [index,loss]
        return log(p(loss|mu,sigma,correlation,sigma_n))
        """
        
        res = 0.0
        if weight is None:
            weight = torch.tensor([1.0]*len(data),device=self.device)
        if data.size(1)==self.num_users: # do in a batch way
            _,index = data[:,:,0].sort()
            val = data[:,:,1].clone()
            for n,idx in enumerate(index):
                val[n,:] = val[n,idx] # rearange the value according to the idx order
            Sigma = self.Covariance()
            distribution = MultivariateNormal(loc = self.mu,covariance_matrix = Sigma)
            res = distribution.log_prob(val)
            res = torch.sum(weight*res)
        else: # do one-by-one
            for i,d in enumerate(data):
                idx = d[:,0].long()
                val = d[:,1]
                Sigma = self.Covariance(idx)
                distribution = MultivariateNormal(loc = self.mu[idx],covariance_matrix = Sigma)
                res += distribution.log_prob(val)*weight[i]

        return res

    def Log_LOO_Predictive_Probability(self,data,weight=None):
        """
        LOO:
        Calculate the Log Leave-One-Out Predictive Probability of the given data
        data: given in the form S x [index,loss]
        return: \sum log(p(y_i|y_{-i},mu,sigma,relation,sigma_n))
        """

        # High efficient algorithm exploiting partitioning
        log_p = 0.0
        if weight is None:
            weight = torch.tensor([1.0]*len(data),device=self.device)
        
        for n,d in enumerate(data):
            idx = d[:,0].long()
            val = d[:,1]
            Sigma_inv = torch.inverse(self.Covariance(idx))
            K_inv_y = (Sigma_inv.mm((val-self.mu[idx]).unsqueeze(1))).squeeze()
            for i in range(len(idx)):
                mu = val[i]-K_inv_y[i]/Sigma_inv[i,i]
                sigma = torch.sqrt(1/Sigma_inv[i,i])
                dist = Normal(loc = mu,scale = sigma)
                log_p+=dist.log_prob(val[i])*weight[n]
        
        return log_p

    def Parameter_Groups(self):
        raise NotImplementedError("A GPR class must have a function to get parameter groups = [Mpar,Spar]")
    
    def MLE_Mean(self):
        """
        Calculate the weighted mean of historical data
        """
        self.mu = torch.zeros(self.num_users,device=self.device).detach()
        current_epoch = max(self.data.keys())
        cum_gamma = torch.zeros(self.num_users,device=self.device)
        for e in self.data.keys():
            for d in self.data[e]:
                idx = d[:,0].long()
                val = d[:,1]
                self.mu[idx] += (self.gamma**(current_epoch-e))*val
                cum_gamma[idx] += self.gamma**(current_epoch-e)
        
        for g in cum_gamma:
            if g == 0.0:
                g+=1e-6
        self.mu = self.mu/cum_gamma
        return self.mu
    
    def Train(self,lr = 1e-2,llr = 1e-2,max_epoches = 5000,schedule_lr = False,schedule_t = None,schedule_gamma = 0.1,update_mean=False,verbose=True,weight_decay=0.0,**kwargs):
        """
        Train hyperparameters(Covariance,noise) of GPR
        data : In shape as [Group,index,value,noise]
        method : {'MML','LOO','NNP'}
            MML:maximize log marginal likelihood
            LOO:maximize Leave-One-Out cross-validation predictive probability 
        """
        
        matrix_params,sigma_params = self.Parameter_Groups()
        optimizer = torch.optim.Adam([{'params':matrix_params,'lr':lr},
                                    {'params':sigma_params,'lr':llr}], lr=lr,weight_decay=weight_decay)
        if schedule_lr:
            lr_scd = torch.optim.lr_scheduler.MultiStepLR(optimizer,schedule_t,gamma = schedule_gamma)

        if update_mean:
            self.mu = self.MLE_Mean()
            #print(self.mu)
        current_epoch = max(self.data.keys())
        # prev_loss = np.inf
        # loss_changes = []
        for epoch in range(max_epoches):
            self.zero_grad()
            training_data = torch.cat([self.data[e] for e in self.data.keys()],dim=0)
            weight = []
            for e in self.data.keys():
                weight += [self.gamma**(current_epoch-e)]*len(self.data[e])
            if self.loss_type == 'LOO':
                loss = -self.Log_LOO_Predictive_Probability(training_data,weight=torch.tensor(weight,device=self.device))
            elif self.loss_type == 'MML':
                loss = -self.Log_Marginal_Likelihood(training_data,weight=torch.tensor(weight,device=self.device))
            else:
                raise RuntimeError("Not supported training method!!")
            loss.backward()
            optimizer.step()
            if (epoch+1)%50==0 and verbose:
                print("Train_Epoch:{}\t|Sigma:{:.4f}\t|Loss:{:.4f}".format(epoch+1,torch.mean(torch.diagonal(self.Covariance())).detach().item(),loss.item()))
                # loss_changes.append(np.abs(loss.item()-prev_loss))
                # if len(loss_changes)>=3 and np.max(loss_changes[-3:])<5.0:# train until converge
                #     break
                # prev_loss = loss.item()
            if schedule_lr:
                lr_scd.step()
                
        return loss.item()
    
    def Predict_Loss(self,data,priori_idx,posteriori_idx):
        for p in priori_idx:
            if p in posteriori_idx:
                posteriori_idx.remove(p) # do not predict the measured idx
        mu_p,sigma_p = self.Posteriori(data[priori_idx,:])

        pdist = MultivariateNormal(loc = mu_p[posteriori_idx],covariance_matrix = sigma_p[posteriori_idx,:][:,posteriori_idx])
        predict_loss = -pdist.log_prob(torch.tensor(data[posteriori_idx,1],device=self.device,dtype=torch.float))
        predict_loss = predict_loss.detach().item()
        return predict_loss,mu_p,sigma_p
        
    def Select_Clients(self,number=10,epsilon = 0.0,weights = None,dynamic_C=False,dynamic_TH=0.0):
        """
        Select the clients which may lead to the maximal loss decrease
        Sequentially select the client and update the postieriori
        """
        def max_loss_decrease_client(client_group,Sigma,weights = None):
            Sigma_valid = Sigma[:,client_group][client_group,:]
            Diag_valid = self.discount[client_group]/torch.sqrt(torch.diagonal(Sigma_valid)) # alpha_k/sigma_k
            if self.rho is not None: # consider the system stat for each client
                Diag_valid = Diag_valid/(self.consumption[client_group]**self.rho) # a_k = a*beta_k/rho_k
            
            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                # sum_i Sigma_ik*p_i
                total_loss_decrease = torch.sum(torch.tensor(weights[client_group],device=self.device,dtype=torch.float).reshape([len(client_group),1])*Sigma_valid,dim=0)*Diag_valid
            mld,idx = torch.max(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = client_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx])

            return selected_idx,p_Sigma,mld.item()

        prob = np.random.rand(1)[0]
        if prob<=epsilon:
            # use epsilon-greedy and random selection
            return np.random.choice(range(self.num_users), number, replace=False)
        else:
            Sigma = self.Covariance()
            remain_clients = list(range(self.num_users))
            selected_clients = []
            for i in range(number):  
                idx,Sigma,total_loss_decrease = max_loss_decrease_client(remain_clients,Sigma,weights)
                if dynamic_C and -total_loss_decrease<dynamic_TH:
                    break
                selected_clients.append(idx)
                remain_clients.remove(idx)
            
            return selected_clients
    

class Kernel_GPR(GPR):
    """
    A GPR class with covariance defined by a kernel function

    Parameters:
        Projection.PMatrix: A Matrix that projects index (in a one-hot vector form)
                            into a low-dimension space. 
                            In fact each column of this matrix corresponds to the location 
                            of that user in the low-dimension space.
        Kernel.sigma_f: Diagonal of covariance matrix, which reveals the uncertainty 
                        priori on each user.We assume the same uncertainty before sampling.
         
        Total number of parameters is num_users x dimension + 2
    """

    
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 0.01,gamma=1.0,rho=None,device = torch.device('cpu'),dimension = 10,kernel = SE_Kernel,**Kernel_Arg):
        class Index_Projection(torch.nn.Module):
            """
            Module that project an index(an int between 0 and num_users-1) to a dimension-D space
            """
            def __init__(self, num_users,dimension=10):
                super(Index_Projection, self).__init__()
                # Normalize the initialization so that the mean of ||x|| is 1
                self.PMatrix = Parameter(torch.randn(dimension,num_users)/np.sqrt(dimension))
                # self.PMatrix = Parameter(torch.ones(dimension,num_users))
            def forward(self,i):
                """
                Return a column vector as the location in the dimension-D space
                """
                return self.PMatrix[:,i]
        super(Kernel_GPR, self).__init__(num_users,loss_type,init_noise,gamma,rho,device)
        self.Projection = Index_Projection(num_users,dimension)
        self.Kernel = kernel(device=device,**Kernel_Arg)


    def Set_Parameters(self,mu=None,proj=None,sigma = None,noise = None):
        if mu is not None:
            self.mu = mu
        if proj is not None:
            self.Projection.PMatrix.data = proj
        if sigma is not None:
            self.Kernel.sigma_f.data = sigma
        if noise is not None:
            self.noise = noise
        

    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix at the given indexes
        """
        if ids is None:
            # Calculate the covariance matrix of all users by default
            ids = list(range(self.num_users))
        xs = self.Projection(ids)
        return self.Kernel(xs)+(self.noise**2)*torch.eye(len(ids),device=self.device)
        

        
    def Parameter_Groups(self):
        proj_parameters = [self.Projection.PMatrix,]
        sigma_parameters = [self.Kernel.sigma_f,] if hasattr(self.Kernel,'sigma_f') else None
        return proj_parameters,sigma_parameters


class Clustered_Kernel_GPR(Kernel_GPR):
    """
    A clustered version of Kernel GPR to avoid high memory overhead. The new version will be 
    a extension of the vanilla Kernel GPR with a clustering algorihm (K-means) and 
    a new selection strategy (bi-level selection).
    """
    def __init__(self,num_users,num_clusters=None,clustering_th=None,loss_type = 'LOO',init_noise = 0.01,gamma=1.0,rho=None,device = torch.device('cpu'),dimension = 10,kernel = SE_Kernel,**Kernel_Arg):
        super(Clustered_Kernel_GPR, self).__init__(num_users,loss_type,init_noise,gamma,rho,device,dimension,kernel,**Kernel_Arg)
        if num_clusters is not None:
            self.num_clusters = num_clusters
            self.max_clusters = num_clusters
        else: # initialize as the number of users
            self.num_clusters = num_users
            self.max_clusters = None

        self.clustering_th = clustering_th
        self.client_cluster,self.centroids = self.Clustering(self.clustering_th,False)

        # calculate discount for each cluster instead of each client
        self.discount = torch.ones(num_clusters,device=device).detach()

        
    def Reset_Discount(self):
        self.discount = torch.ones(self.num_clusters,device=self.device).detach()
    
    def Update_Discount(self,index,factor=0.9):
        cluster_idx = list(set(self.client_cluster[index].cpu().numpy().tolist()))
        self.discount[cluster_idx]*=factor

    def Covariance(self,user_ids=None,cluster_ids=[]):
        """
        Return the Covariance Matrix at between [user_ids,cluster_ids]
        """
        if user_ids is None:
            # Calculate the covariance matrix of all users by default
            user_ids = list(range(self.num_users))
        if cluster_ids is None:
            cluster_ids = list(range(self.num_clusters))

        
        xs = []
        all_user_xs = self.Projection(list(range(self.num_users)))
        if len(user_ids)>0:
            user_xs = all_user_xs[:,user_ids]
            # selected_user_xs = user_xs[:,user_ids]
            xs.append(user_xs)

        if len(cluster_ids)>0:
            #cluster_xs = self.cluster_embeddings[:,cluster_ids]
            cluster_xs = []
            for i in cluster_ids:
                # calculate the centroid embeddings as the mean of the client embeddings in the cluster
                cluster_xs.append(torch.mean(all_user_xs[:,self.client_cluster==i],dim=1,keepdim=True))
            cluster_xs = torch.cat(cluster_xs,dim=1)
            xs.append(cluster_xs)
        
        
        xs = torch.cat(xs,dim=1)
        return self.Kernel(xs)+(self.noise**2)*torch.eye(len(user_ids)+len(cluster_ids),device=self.device)
    
    
    def Train(self,**kwargs):
        loss = super(Clustered_Kernel_GPR, self).Train(**kwargs)
        self.client_cluster,self.centroids = self.Clustering(self.clustering_th,True)# Re-cluster after training the embeddings
        return loss

        
    def Clustering(self,dynamic_clustering_th=None,prev_init=True):
        """
        Use normalized user embeddings for clustering
        """
        def get_cluster_centroid(cluster_data):
            """
            Calculate the mean of the data in each cluster
            """
            centroid = torch.mean(cluster_data,dim=0,keepdim=True)
            return centroid

        def quantize(input,centroids):
            """
            assign data to the closest centroids given centroids
            return: 
            distance: The distance of each data to its assigned centroid
            cent_idx: The index of the assigned centroid to each data
            """
            if len(input.shape)==1:
                input = input.reshape([1,-1]) # make it as N x D
            
            
            distance_matrix = []
            for i in range(len(centroids)):
                distance_matrix.append(torch.norm(input-centroids[i,:].reshape([1,-1]),p=2,dim=1,keepdim=True))
            distance_matrix = torch.cat(distance_matrix,dim=1)
            distance,cent_idxs = torch.min(distance_matrix,dim=1)
            return distance,cent_idxs

        
        def kmeans(data,initial_centroids=None,num_centroids=None,tolerance=1e-4,max_iter=100):
            """
            Perform k_means given initial centroids
            """
            if initial_centroids is None:
                assert num_centroids is not None, "[Error]: Initial_centroids is None while num_centroids id not given in k_means!"
                indices = np.random.choice(len(data), num_centroids, replace=False)
                centroids = data[indices,:]
            else:
                centroids = initial_centroids
            
            for _ in range(max_iter):
                new_centroids = deepcopy(centroids)
                distance,cent_idxs = quantize(data,centroids) # cluster
                for c in range(len(centroids)):
                    new_centroids[c] = get_cluster_centroid(data[cent_idxs==c]) # mean
                max_move = torch.max(torch.norm(new_centroids-centroids,p=2,dim=1))
                if max_move<tolerance:
                    break
                centroids = new_centroids
            
            return cent_idxs,centroids,distance

        def incremental_kmeans(data,distortion,max_clusters=None,**kargs):
            """
            perform incremental k_means given distortion tolerance
            """
            if max_clusters is None:# unlimited number for clusters
                max_clusters = np.inf
            centroids = get_cluster_centroid(data) # initialize data
            val_distance, cent_idxs = quantize(data,centroids)
            while True:
                #mean_dis = torch.mean(val_distance,dim=0)
                max_dis,max_idx = torch.max(val_distance,dim=0)

                if max_dis>distortion and len(centroids)<max_clusters: # keep adding new centroids until the constraint is satisfied
                    centroids = torch.cat([centroids,data[max_idx,:].reshape([1,-1])],dim=0)
                    cent_idxs,centroids,val_distance = kmeans(data,centroids,kargs)
                else:
                    break
            return cent_idxs,centroids,val_distance

        user_xs = self.Projection(list(range(self.num_users))).detach() 
        user_xs = (user_xs/torch.norm(user_xs,p=2,dim=0,keepdim=True)).transpose(0,1) # normalize the embeddings and transpose to be in shape [user_num,dim]
        if dynamic_clustering_th is None: # use fixed number of clusters
            if prev_init:
                init_centroids = torch.zeros_like(self.centroids,device=self.device)
                for i in range(self.num_clusters):
                    init_centroids[i]=get_cluster_centroid(user_xs[self.client_cluster==i])
            else:
                init_centroids = None
            client_cluster,centroids,dist = kmeans(user_xs,init_centroids,num_centroids=self.num_clusters)
        
        else: # use dynamic number of clustering
            client_cluster,centroids,dist = incremental_kmeans(user_xs,distortion=dynamic_clustering_th,max_clusters = self.max_clusters)
            self.num_clusters = len(centroids)
            print("Dynamic clustering finished, get %d clusters"%self.num_clusters)

        print("Clustering finished with mean quantization distance {:.2f}".format(torch.mean(dist)))

        return client_cluster,centroids
        

    
    def Select_Clients(self,number=10,epsilon = 0.0,weights = None,dynamic_C=False,dynamic_TH=0.0):
        """
        Select the clients which may lead to the maximal loss decrease
        Sequentially select the client and update the postieriori
        """
        def max_loss_decrease_cluster(cluster_group,Sigma,weights = None):
            Sigma_valid = Sigma[:,cluster_group][cluster_group,:]
            Diag_valid = self.discount[cluster_group]/torch.sqrt(torch.diagonal(Sigma_valid)) # alpha_k/sigma_k
            if self.rho is not None: #calculate the system stat for each cluster
                cluster_system_stat = torch.tensor([torch.sum(self.client_cluster==i)/torch.sum(1./self.consumption[self.client_cluster==i]) for i in cluster_group]).to(Diag_valid)
                Diag_valid = Diag_valid/(cluster_system_stat**self.rho) # a_k = a*beta_k/rho_k

            if weights is None:
                total_loss_decrease = torch.sum(Sigma_valid,dim=0)*Diag_valid
            else:
                # sum_i Sigma_ik*p_i
                total_loss_decrease = torch.sum(weights[cluster_group].reshape([len(cluster_group),1])*Sigma_valid,dim=0)*Diag_valid

            mld,idx = torch.max(total_loss_decrease,0)
            idx = idx.item()
            selected_idx = cluster_group[idx]
            p_Sigma = Sigma-Sigma[:,selected_idx:selected_idx+1].mm(Sigma[selected_idx:selected_idx+1,:])/(Sigma[selected_idx,selected_idx])

            return selected_idx,p_Sigma,mld.item()
        
        prob = np.random.rand(1)[0]
        if prob<=epsilon:
            # use epsilon-greedy and random selection
            return np.random.choice(range(self.num_users), number, replace=False)
        else:
            if weights is not None:
                weights = torch.tensor(weights,device=self.device,dtype=torch.float)
                cluster_weights = []
                for i in range(self.num_clusters):
                    cluster_weights.append(torch.sum(weights[self.client_cluster==i]))
                cluster_weights = torch.tensor(cluster_weights,device=self.device,dtype=torch.float)
            else:
                cluster_weights = None
            Sigma = self.Covariance([],None) # covariance among clusters
            remain_clusters = list(range(self.num_clusters))
            selected_clusters = []
            for i in range(number):
                if len(remain_clusters)==0:
                    break  
                idx,Sigma,total_loss_decrease = max_loss_decrease_cluster(remain_clusters,Sigma,cluster_weights)
                if dynamic_C and -total_loss_decrease<dynamic_TH:
                    break
                selected_clusters.append(idx)
                remain_clusters.remove(idx)
            
            selected_clients = []
            for i in selected_clusters: # select one client from each cluster randomly
                if torch.sum(self.client_cluster==i)>0:
                    if self.rho is not None:
                        p = 1./self.consumption[self.client_cluster==i] # p_k~1/U_k
                        p = (p/torch.sum(p)).cpu().numpy()
                        selected_clients.append(np.random.choice(np.arange(self.num_users)[self.client_cluster.cpu().numpy()==i],p=p))
                    else:
                        selected_clients.append(np.random.choice(np.arange(self.num_users)[self.client_cluster.cpu().numpy()==i]))
            return selected_clients
        
    
        
class Matrix_GPR(GPR):
    """
    A GPR class with covariance defined by a positive definite matrix Sigma

    Parameters:
        Lower: Elements of the lower triangular matrix L except the diagonal
        
        Diagonal: |Diagonal| will be the diagonal elements of L 

        The Covariance Matrix Priori is computed as LL'.
        The total number of parameters is (num_users*num_users+num_users)//2+1
    """
    def __init__(self,num_users,loss_type = 'LOO',init_noise = 0.01,gamma=1.0,rho=None,device = torch.device('cpu')):
        super(Matrix_GPR, self).__init__(num_users,loss_type,init_noise,gamma,rho,device=device)
        # Lower Triangular Matrix L Elements without diagonal elements
        self.Lower = Parameter(torch.zeros((num_users*num_users-num_users)//2))
        self.index = torch.zeros((num_users*num_users-num_users)//2,dtype = torch.long)
        n = 0
        for i in range(num_users):
            for j in range(num_users):
                if j<i:# an lower triangular matrix 
                    self.index[n]=i*num_users+j
                    n+=1
                else:
                    break
        # Diagonal elements of L
        self.Diagonal = Parameter(torch.ones(num_users))

    def Set_Parameters(self,mu=None,diag=None,noise = None,lower = None):
        if mu is not None:
            self.mu.copy_(mu)
        if diag is not None:
            self.Diagonal.data = diag
        if noise is not None:
            self.noise = noise
        if lower is not None:
            self.Lower.data = lower


    def Covariance(self,ids = None):
        """
        Return the Covariance Matrix according to Lower Trangular Matrix
        """
        L = torch.zeros(self.num_users*self.num_users,device=self.device)
        L.scatter_(0,self.index,self.Lower)
        L = L.reshape([self.num_users,self.num_users])
        L = L+torch.abs(torch.diag(self.Diagonal))#Now we get L
        # Sigma = LL'
        Sigma = L.mm(L.transpose(0,1))
        if ids is None:
            # Return the covariance matrix of all users by default
            return Sigma+(self.noise**2)*torch.eye(self.num_users,device=self.device)
        else:
            return Sigma[ids,:][:,ids]+(self.noise**2)*torch.eye(len(ids),device=self.device)

    def Parameter_Groups(self):
        matrix_parameters = [self.Lower,self.Diagonal]
        sigma_parameters = None
        return matrix_parameters,sigma_parameters



class FedCor_Selector(Selector):
    """
    FedCor/Clustered FedCor
    Args:
    clustered: If True, use Clustered FedCor, else use FedCor
    rho: If not None, consider systematic information when selecting
    """
    def __init__(self, total_client_num, weights=None, clustered = True,
                 kernel = 'Poly', poly_norm = False, dimension = 15, noise=0.01, discount = 0.9, 
                 dynamic_C=False,dynamic_TH=0.0,greedy_epsilon = 0.0,
                 num_cluster = 50, clustering_th = 0.1,#user_batch=50,cluster_batch=0,
                 train_method= 'MML',GPR_gamma=0.95,rho = None,
                 gpr_begin=0,warmup=20,GPR_interval=20,update_mean=True, 
                 gpr_gpu=None,**kwargs) -> None:
        super().__init__(total_client_num, weights)
        device = 'cuda:' + gpr_gpu if gpr_gpu else 'cpu'
        if clustered:
            if kernel=='Poly':
                self.gpr = Clustered_Kernel_GPR(total_client_num,num_clusters=num_cluster,clustering_th=clustering_th,
                                                loss_type= train_method,init_noise=noise,gamma=GPR_gamma,rho=rho,device=device,
                                                dimension = dimension,kernel=Poly_Kernel,order = 1,poly_norm = poly_norm)
            elif kernel=='SE':
                self.gpr = Clustered_Kernel_GPR(total_client_num,num_clusters=num_cluster,clustering_th=clustering_th,
                                                loss_type= train_method,init_noise=noise,gamma=GPR_gamma,rho=rho,device=device,
                                                dimension = dimension,kernel=SE_Kernel)
            else:
                raise RuntimeError("Clustered GPR must use kernel!")
        else:
            if kernel=='Poly':
                self.gpr = Kernel_GPR(self.total_client_num,loss_type=train_method,init_noise=noise,
                                gamma=GPR_gamma,rho=rho,device=device, dimension=dimension,kernel=Poly_Kernel,order=1,poly_norm=poly_norm)
            elif kernel=='SE':
                self.gpr = Kernel_GPR(self.total_client_num,loss_type=train_method,init_noise=noise,
                                gamma=GPR_gamma,rho=rho,device=device, dimension=dimension,kernel=SE_Kernel)
            else: # Matrix optimization directly
                self.gpr = Matrix_GPR(self.total_client_num,loss_type=train_method,init_noise=noise,gamma=GPR_gamma,rho=rho,device=device)
        
        self.gpr.to(device)
        self.gpr_begin = gpr_begin
        self.gpr_warmup = warmup
        # self.user_batch=user_batch
        # self.cluster_batch=cluster_batch
        self.train_interval = GPR_interval
        self.update_mean = update_mean
        self.gpr_beta = discount
        self.dynamic_C = dynamic_C
        self.dynamic_TH = dynamic_TH
        self.epsilon = 1.0
        self.greedy_epsilon = greedy_epsilon
        self.update_discount = False
        self.round = 0

    def select(self, select_num=0, **kwargs):
        # if epsilon is set as 1.0, then this selection will be the same as random selection
        idxs_users = self.gpr.Select_Clients(select_num,self.epsilon,self.weights,self.dynamic_C,self.dynamic_TH)
        return idxs_users
    
    def stat_update(self, epoch=None, selected_clients=None, stat_info = None, sys_info = None, server = None, verbose=True, **kwargs):
        """
        stat_info: should be the loss changes of all clients in this round
        sys_info: should be the estimated training + communication cost of all clients 
        """
        if epoch is not None:
            self.round = epoch
        if stat_info is not None and self.round >= self.gpr_begin: # sample and store training samples
            if server is not None and (self.round <= self.gpr_warmup or self.round%self.train_interval==0):
                print("Training with Random Selection For GPR Training:")
                m = max(int(server.train_frac * server.num_users), 1)
                random_idxs_users = np.random.choice(range(server.num_users), m, replace=False)
                new_model = server.train_idx(idxs_users = random_idxs_users)
                new_stat_info = server.val(new_model)
                stat_info = new_stat_info["val_loss"]-stat_info["val_loss"]
                self.gpr.Update_Training_Data([np.arange(self.total_client_num),],[stat_info,],epoch=self.round)
        if self.round >= self.gpr_warmup: # update annealing factor
            self.epsilon = self.greedy_epsilon
            if selected_clients is not None:
                self.gpr.Update_Discount(selected_clients,self.gpr_beta)
        if self.round == self.gpr_warmup or (self.round>self.gpr_warmup and self.round%self.train_interval==0): # training round for GP
            print("Training GPR")
            self.gpr.Train(lr = 1e-2,llr = 1e-2,max_epoches=5000 if self.round == self.gpr_warmup else 500,schedule_lr=False,update_mean=self.update_mean,weight_decay=1e-4,verbose=verbose)
            self.gpr.Reset_Discount() # reset discount after each training
        
        if sys_info is not None:
            self.gpr.Update_System_Stat(np.arange(self.total_client_num),sys_info["train_times"]) # update systematic information


# if __name__=='__main__':
#     num_users = 30
#     gpr = Kernel_GPR(num_users,dimension = 1,init_noise=0.01,kernel=SE_Kernel,l = 1.0)

#     pmatrix = np.zeros([1,num_users])
    
#     pmatrix[0,:] = np.arange(num_users,dtype = np.float)/5
#     gpr.set_parameters(proj=torch.tensor(pmatrix))
#     sel = gpr.Select_Clients(number=3,discount_method='time',verbose=True)
#     print(sel)
#     plt.show()

    



