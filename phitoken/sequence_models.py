import numpy as np
import scipy.stats
import scipy.special
import scipy.signal

import heapq

logfn = np.log

def _clogc(c):
    if c == 0:
        return 0.0
    else:
        return c*logfn(c)


class OnlineLinearRegression(object):

    def __init__(
        self,
        n_features,
        initial_coeff=None,
        min_alpha=1e-4,
        step_size=0.01,
    ):
        self.n_features = n_features 

        if initial_coeff is None:
            self._coeff = np.zeros(n_features)
        else:
            self._coeff = initial_coeff.copy()
        
        self.min_alpha = min_alpha

        self._mu_y = 0.0
        self._mu_x = np.zeros(n_features)
        self._var_gx = np.ones(n_features)

        self._resid_var = 0.0

        self.step_size = step_size
        self._t = 1


    def predict(self, x):
        dx = x - self._mu_x
        return self._mu_y + np.dot(dx, self._coeff)


    def update(self, x, y):
        alpha = max(self.min_alpha, 1.0/self._t)

        #first update the historical average values
        self._mu_y = self._mu_y*(1-alpha) + alpha*y
        self._mu_x = self._mu_x*(1-alpha) + alpha*x

        #get predictions and residuals
        y_pred = self.predict(x)
        residual = y - y_pred

        self._resid_var = self._resid_var*(1.0-alpha) + alpha*residual**2

        #calculate the unscaled update
        dx = x - self._mu_x
        unscaled_grad = residual*dx

        #update the delta scale factors
        self._var_gx = self._var_gx*(1-alpha) + alpha*unscaled_grad**2

        #and turn it into a step size scale factor
        tau = np.sqrt(1e-6 + self._var_gx)

        #then take a step in coefficient space
        step_size_noise = np.random.random(size=(self.n_features,))
        self._coeff += (self.step_size*residual)*(dx/tau)*step_size_noise

        self._t += 1


class MarkovNode(object):

    def __init__(
            self,
            parent,
            symbol,
            N,
            gamma,
            vocab_prior,
            branch_depth=1,
    ):
        self.parent = parent
        self.symbol = symbol
        self.children = {}
        self.N = N
        self.gamma = gamma
        self.vocab_prior = vocab_prior
        self.branch_depth = branch_depth

        self.ent_sums = []
        self.singleton_counts = []
        for depth in range(self.branch_depth):
            cardinality_prior = vocab_prior**depth
            self.ent_sums.append(gamma*logfn(gamma/cardinality_prior))
            self.singleton_counts.append(0)


    def get(self, item, fallback=None):
        if item in self.children:
            return self.children[item]
        else:
            return fallback

    def __getitem__(self, index):
        return self.children[index]
    
    def update(self, seq, amount=1, node_list=[]):
        path = [self]
        for symbol in seq:
            current = path[-1]
            next = current.get(symbol)
            if next is None:
                next = MarkovNode(
                    parent=current,
                    symbol=symbol,
                    N=0,
                    gamma=self.gamma,
                    vocab_prior=self.vocab_prior,
                    branch_depth=self.branch_depth,
                )
                current.children[symbol] = next
                node_list.append(next)
            path.append(next)
        
        #update the continuation entropy estimates for each depth
        for depth_idx in range(self.branch_depth):
            depth = depth_idx + 1
            for node_idx, node in enumerate(path[:-depth]):
                downstream_node = path[node_idx+depth]
                sum_delta = _clogc(downstream_node.N + amount) - _clogc(downstream_node.N)
                node.ent_sums[depth_idx] += sum_delta
                if downstream_node.N == 0:
                    #this node is fresh and so will become a singleton on the count update step.
                    node.singleton_counts[depth_idx] += 1
                elif downstream_node.N == 1:
                    #this used to be a singleton and is about to get updated to a double
                    node.singleton_counts[depth_idx] -= 1

        #then increment the counts for all nodes along the path
        for node in path:
            node.N += amount
    
    # def update(self, seq, amount=1, node_list=[]):
    #     self.N += amount
    #     if len(seq) > 0:
    #         next_symbol = seq[0]
    #         child = self.children.get(next_symbol)
    #         if child is None:
    #             #add the new child node in
    #             child = MarkovNode(
    #                 parent=self,
    #                 symbol=next_symbol,
    #                 N=0,
    #                 gamma=self.gamma, 
    #                 vocab_prior=self.vocab_prior
    #             )
    #             node_list.append(child)
    #             self.children[next_symbol] = child

    #         trunc_seq = seq[1:]
    #         child.update(trunc_seq, amount=amount, node_list=node_list)
            
    #         #manage the online update for entropy calculations
    #         deltaclogc = _clogc(child.N) - _clogc(child.N-amount)
    #         self.ent_sum += deltaclogc

    def continuation_entropies(self):
        neff = self.neff
        ents = [np.log(neff) - esum/neff for esum in self.ent_sums]
        return ents


    @property
    def p(self):
        cprob = 1.0
        cnode = self
        while not (cnode.parent is None):
            pnode = cnode.parent
            v = pnode.vocab_prior
            cond_prob =  (cnode.N + pnode.gamma/v)/(pnode.N + pnode.gamma)
            cprob *= cond_prob
            cnode = pnode
        return cprob

    @property
    def neff(self):
        return self.N + self.gamma


def deletion_score(node):
    parent = node.parent
    if parent is None:
        return np.inf
    v = parent.vocab_prior
    distortion = logfn((parent.gamma + node.N)/(parent.gamma + node.N/v))
    return -1*node.neff*distortion


class MarkovModel(object):

    def __init__(
        self, 
        order,
        gamma=1.0,
        vocab_prior=1000,
        max_nodes=2**24,
        start_symbol="",
        stop_symbol="",
        entropy_branch_depth=1,
    ):
        assert entropy_branch_depth >= 0

        self.root = MarkovNode(
            parent=None,
            symbol=None,
            N=0,
            gamma=gamma, 
            vocab_prior=vocab_prior,
            branch_depth=entropy_branch_depth,
        )
        self.order = order
        self.gamma = gamma
        self.vocab_prior = vocab_prior
        self.max_nodes = max_nodes
        self.start_symbol = start_symbol
        self.stop_symbol = stop_symbol
        self.node_list = [self.root]

        self.xent_predictors = [
            OnlineLinearRegression(
                n_features=5,
                step_size=0.01,
                #initial_coeff=np.asarray([0.75, -1.25, 1.0, 2.0])
            )
            for i in range(1)#self.order)
        ]

    def prune(self, count_threshold=1):
        new_node_list = []
        for node in self.node_list:
            if node.N > count_threshold:
                new_node_list.append(node)
            else:
                parent = node.parent

                #delete the node from the child list of the parent node
                del_node = parent.children.pop(node.symbol)
                assert node is del_node #sanity check            
                #reduce parent N by the deletion count and add it to gamma
                #this way the denominator for probability estimates don't change
                #note that updating this way only works if parent nodes always precede their children in the node_list ordering
                parent.gamma = parent.gamma + node.N
                parent.N = parent.N - node.N

        self.node_list = new_node_list

    def update(
            self, 
            seq, 
            amount=1,
            inject_start=True,
            inject_stop=True,
            auto_prune=True,
        ):
        #import pdb; pdb.set_trace()

        subseq_len = self.order + 1
        if inject_start:
            subseq = [self.start_symbol] + list(seq[:subseq_len-1])
            self.root.update(subseq, amount=amount, node_list=self.node_list)

        for i in range(len(seq) - self.order -1):
            subseq = seq[i:i+subseq_len]
            next_symbol = seq[i+subseq_len]

            condition_nodes = self.get_condition_candidates(subseq, max_lookback=self.order)
            condition_feats = self.calculate_condition_features(condition_nodes)
            
            p_of_actual = []
            for context_depth, cnode in enumerate(condition_nodes):
                extension_node = cnode.get(next_symbol)
                if extension_node is None:
                    extension_count = 0
                else:
                    extension_count = extension_node.N
                vsize = cnode.vocab_prior
                cprob = (extension_count + cnode.gamma/vsize)/cnode.neff
                p_of_actual.append(cprob)
            
            #for xepredictor, cx, cp in zip(self.xent_predictors, condition_feats, p_of_actual):
            for cx, cp in zip(condition_feats, p_of_actual):
                xepredictor = self.xent_predictors[0]
                xepredictor.update(cx, -1.0*logfn(cp))

            self.root.update(subseq, amount=amount, node_list=self.node_list)

        if inject_stop:
            for i in range(len(seq) - self.order, len(seq)):
                partial_seq = list(seq[i:])
                npad = subseq_len - len(partial_seq)
                subseq = partial_seq + [self.stop_symbol for j in range(npad)]
                self.root.update(subseq, amount=amount, node_list=self.node_list)
        
        if auto_prune and len(self.node_list) > self.max_nodes:
            self.prune()

    def walk_tree(self, seq):
        leaf = self.root
        for item in seq:
            leaf = leaf.get(item)
            if leaf is None:
                break
        return leaf

    def seq_count(self, seq):
        leaf = self.walk_tree(seq)
        if leaf is None:
            return 0
        return leaf.N

    def get_condition_candidates(
            self, 
            seq,
            max_lookback
        ):
        conditions = [self.root]

        for lookback in range(1, min(len(seq), max_lookback)):
            prefix_seq = seq[-lookback:]
            candidate_node = self.walk_tree(prefix_seq)
            if (candidate_node is None):
                break

            conditions.append(candidate_node)
        
        return conditions

    def calculate_condition_features(self, conditions):
        condition_vecs = []
        for cnode in conditions:
            cN = cnode.N
            cN = max(cN, 1)
            c_clogc = cnode.ent_sums[0]
            feats = np.asarray([
                logfn(cN),
                c_clogc/cN,
                1.0/cN,
                (c_clogc/cN)**2,
                cnode.singleton_counts[0]/cN,
            ])
            condition_vecs.append(feats)
        return condition_vecs

    def calc_probs(
            self, 
            seq,
            inject_start=True,
            inject_stop=True,
            max_lookback=None,
        ):
        if max_lookback is None:
            max_lookback = self.order

        if inject_start:
            if not seq[0] == self.start_symbol:
                seq.insert(0, self.start_symbol)
        
        if inject_stop:
            if not seq[-1] == self.stop_symbol:
                seq.append(self.stop_symbol)

        probs = []
        for i in range(len(seq)):
            prefix = seq[:i]
            condition_nodes = self.get_condition_candidates(prefix, max_lookback=max_lookback)

            n_cond = len(condition_nodes)
            pvec = np.zeros(len(condition_nodes))

            node_xent_estimates = np.zeros(len(condition_nodes))
            condition_feats = self.calculate_condition_features(condition_nodes)

            for idx, cnode in enumerate(condition_nodes):
                #node_xent_estimates[idx] = self.xent_predictors[idx].predict(condition_feats[idx])    
                node_xent_estimates[idx] = self.xent_predictors[0].predict(condition_feats[idx])    

                extension_node = cnode.get(seq[i])
                
                if extension_node is None:
                    extension_count = 0
                else:
                    extension_count = extension_node.N

                vsize = cnode.vocab_prior
                cprob = (extension_count + cnode.gamma/vsize)/cnode.neff
                pvec[idx] = cprob
            
            #weights = calc_ismax_prob(
            #    -1.0*node_xent_estimates, 
            #    np.repeat(0.1, n_cond)
            #)
            weights = scipy.special.softmax(-1.0*node_xent_estimates)
            #weights = scipy.signal.unit_impulse(n_cond, np.argmin(node_xent_estimates))

            extension_prob = np.sum(weights*pvec)
            probs.append(extension_prob)
        return probs


def maxmix_gaussians(
    mu1, sig1,
    mu2, sig2
):
    theta = np.sqrt(sig1**2 + sig2**2)
    p1 = scipy.stats.norm.cdf((mu1-mu2)/theta)
    p2 = scipy.stats.norm.cdf((mu2-mu1)/theta)
    p3 = scipy.stats.norm.pdf((mu1-mu2)/theta)#1.0/(np.sqrt(2*np.pi))*np.exp(-0.5*((mu1-mu2)/theta)**2)
    
    mu_mix = p1*mu1 + p2*mu2 + p3*theta
    
    #second moment
    xsq_mix = p1*(mu1**2 + sig1**2) + p2*(mu2**2 + sig2**2) + p3*theta*(mu1 + mu2)
        
    #turn the second moment into a standard deviation
    sig_mix = np.sqrt(xsq_mix - mu_mix**2)

    return mu_mix, sig_mix

def calc_ismax_prob(
    mu, 
    sig,
):
    if len(mu) == 1:
        return np.ones(1)
    
    opt_val = mu[0]
    opt_sig = sig[0]

    for i in range(1, len(mu)):
        opt_val, opt_sig = maxmix_gaussians(opt_val, opt_sig, mu[i], sig[i])
    
    #prob option minus the estimated max distrib comes out positive
    opt_like = scipy.stats.norm.cdf((mu - opt_val)/np.sqrt(opt_sig**2 + sig**2))
    opt_probs = opt_like/np.sum(opt_like)

    return opt_probs
