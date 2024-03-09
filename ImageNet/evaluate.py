import numpy as np
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy import stats as STS
import cv2

def normalization(saliency):
    #各重要度画像に対して正規化を行う
    min_value = saliency.min(axis=(1, 2, 3),keepdims=True)
    max_value = saliency.max(axis=(1, 2, 3),keepdims=True)
    mask = (saliency - min_value) / (max_value - min_value)
    return mask

def plot_ins_del(threshold, scores,ins_del,name,N,result_path):
    plt.plot(threshold, scores, color="blue")
    plt.fill_between(threshold, np.array(scores)[:], color="lightblue")
    plt.ylim(bottom=0, top=1.01)
    plt.savefig(result_path+f"{ins_del}_{name}_{N}.png")
    plt.close()

class Insertion_Deletion():
    def __init__(self,saliency,test_images,model,label,N):
        self.saliency = saliency
        self.test_images = test_images
        self.model = model
        self.label = label
        self.N = N

    def auc(self,arr):
        """Returns normalized Area Under Curve of the array."""
        return (np.sum(arr) - arr[0] / 2 - arr[-1] / 2) / (np.array(arr).size - 1)
    

    def insertion_deletion_run(self,result_path,name,run=True):
        auc_scores = {'del': [], 'ins': []}
        #どの割合で挿入削除していくのかを決定
        ratio = np.arange(0.0, 1.01, 0.036)
        threshold = np.round(ratio, 2)
        #重要度マップの正規化
        mask = normalization(self.saliency)
        mean = [103.939, 116.779, 123.68]
        sample = self.test_images+mean
        sample=np.where(sample>255,255,sample)
        sample=np.where(sample<0,0,sample)
        sample = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) for img in sample]
    
        print("run_start")
        #全テスト画像を回す
        print("start_insertion")
        ins_del = 'ins'
        for jj in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            insetion_score=[]

            #重要度順に画像のチャンネルを挿入していく
            for i in threshold[::-1]:         
                ins_mask = np.where(mask[jj]>=i,1.,0.)
                mask_images = sample[jj] * ins_mask

                mask_images=cv2.cvtColor(mask_images.astype(np.uint8), cv2.COLOR_RGB2BGR)
                mask_test_images = np.array(mask_images) - mean
                
                preds = self.model.predict(np.expand_dims(mask_test_images,axis=0))
                score = preds[0, self.label[jj]]
                insetion_score.append(score)
              
            insetion_score = np.array(insetion_score)
            score_min = insetion_score[0]
            score_max = insetion_score[-1]
            scores = (insetion_score - score_min) / (score_max - score_min)
            scores = np.maximum(scores,0)
            scores = np.minimum(scores,1)
            if run==False:
                plot_ins_del(threshold, scores,ins_del,name,self.N,result_path)
            auc_scores['ins'].append(self.auc(scores))
        ins_score = np.mean(auc_scores['ins'])
        print("end_insertion")
        print(ins_score)
    
        #全テスト画像を回す
        ins_del = 'del'
        for jj in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            deletion_score=[]

            for i in threshold[::-1]:            
                del_mask = np.where(mask[jj]<=i,1,0)
                mask_images = sample[jj] * del_mask
                mask_images=cv2.cvtColor(mask_images.astype(np.uint8), cv2.COLOR_RGB2BGR)
                mask_test_images = np.array(mask_images) - mean
                if i==0.0:
                    #全てが0値の時は、0の代わりに0.1を入れる
                    mask_test_images = np.full_like(mask_test_images, 0.1)
                
                preds = self.model.predict(np.expand_dims(mask_test_images,axis=0))
                score = preds[0,self.label[jj]]
                deletion_score.append(score)
            
            deletion_score = np.array(deletion_score)

            score_max = deletion_score[0]
            score_min = deletion_score[-1]
            scores = (deletion_score - score_min) / (score_max - score_min)
            scores = np.maximum(scores,0)
            scores = np.minimum(scores,1)
            if run==False:
                plot_ins_del(threshold, scores,ins_del,name,self.N,result_path)
            auc_scores['del'].append(self.auc(scores))
        del_score = np.mean(auc_scores['del'])
        print("end_deletion")
        print(del_score)
        #結果をテキストに保存
        if run:
            with open(result_path+f"result_ins_del_{self.N}.txt", "w") as o:
                print(f"insertion:{ins_score}\n", file=o)
                print(f"deletion:{del_score}\n", file=o)

class Adcc():
    def __init__(self,saliency,explainer,model,test_images,test_labels,maskname,p_mask,N):
        self.saliency = saliency
        self.explainer = explainer
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels
        self.maskname = maskname
        self.p_mask = p_mask
        self.N = N


    def average_drop(self,explanation_map, model, predict, class_idx):

        out_on_exp = model(np.expand_dims(explanation_map,axis=0))

        confidence_on_inp = predict

        confidence_on_exp = out_on_exp[:,class_idx][0]

        return np.maximum((confidence_on_inp-confidence_on_exp),0)/confidence_on_inp

    def coherency(self,saliency_map, explanation_map, model,class_idx,saliency_map_B, out):
        
        A, B = saliency_map, saliency_map_B

        '''
        # Pearson correlation coefficient
        # '''
        Asq, Bsq = A.reshape(1,-1)[0], B.reshape(1,-1)[0]

        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        #nanがある場合計算ができないので、それの対応
        if np.isnan(Asq).any() or np.isnan(Bsq).any():
            y = 1e-10
        elif Asq.max()==0:
            y = 1e-10
        elif Bsq.max()==0:
            y = 1e-10
        else:     
            y, _ = STS.pearsonr(Asq, Bsq)
            y = (y + 1) / 2
        
        return y,A,B

    def complexity(self,saliency_map):
        return abs(saliency_map).sum()/(saliency_map.shape[0]*saliency_map.shape[1]*saliency_map.shape[2])

    def ADCC(self, image, saliency_map, explanation_map, model, saliency_map_B, class_idx):
        out = model(image)[:,class_idx][0]
        avgdrop = self.average_drop(explanation_map, model, predict=out, class_idx=class_idx)
        
        coh,A,B=self.coherency(saliency_map,explanation_map, model,class_idx, saliency_map_B, out=out)
        com=self.complexity(saliency_map)
        
        adcc = 3 / (1/coh + 1/(1-com) +1/(1-avgdrop))
        #if np.isnan(adcc).any():
            #pdb.set_trace()
        return adcc

    #実際にADCCを走らせるコード(ADCCはもう一度手法にかける必要あり)
    def adcc_run(self,mode, result_path, mask_path,run=True):

        norm_saliency = normalization(self.saliency)
        #camを元画像に適用して、もう一度手法にかける
        mean = [103.939, 116.779, 123.68]
        sample = self.test_images+mean
        sample=np.where(sample>255,255,sample)
        sample=np.where(sample<0,0,sample)
        for i in range(self.test_images.shape[0]):
            sample[i] = cv2.cvtColor(sample[i].astype(np.uint8), cv2.COLOR_BGR2RGB)
        test_images2 = sample*norm_saliency
        for i in range(test_images2.shape[0]):
            test_images2[i]=cv2.cvtColor(test_images2[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        explanation_map = np.array(test_images2) - mean

        self.explainer.load_masks(mask_path,self.maskname,p1=self.p_mask)

        saliency_list=[]
        for i in tqdm(range(self.test_images.shape[0]), desc="Processing"):
            score_list,masks,base_mask = self.explainer.forward(explanation_map[i])
            score=np.expand_dims(score_list[:,self.test_labels[i]],axis=(-1,-2,-3))
            saliency = np.mean(masks*score,axis=0)
            if mode == 'RaCF_GradCAM':
                #GradCAMによって評価されない領域を、評価領域の最低値に合わせる
                #たまにsaliencyが0の時があるため、場合わけ
                if saliency.min()==saliency.max():
                    saliency_list.append(saliency)
                    continue
                aa=np.reshape(saliency,[-1])
                bb=np.sort(aa,axis=-1)
                min_num = np.where(bb>0)[0][0]
                saliency=np.where(saliency<=0,bb[min_num],saliency)
            saliency_list.append(saliency)
        saliency_B=np.array(saliency_list)
        if run:
            with open(result_path+f"saliency_B{self.N}.pickle","wb") as aa:
                pickle.dump(saliency_B, aa,protocol=4)

        norm_saliency_B = normalization(saliency_B)

        adcc_list=[]
        for i in range(self.test_images.shape[0]):
            score = self.ADCC(self.test_images[i:i+1], norm_saliency[i], explanation_map[i], self.model, norm_saliency_B[i], self.test_labels[i])
            adcc_list.append(score)
        all_scores = np.mean(adcc_list)
        print(all_scores)
        if run:
            with open(result_path+f"result_adcc{self.N}.txt", "w") as o:
                print(f"adcc:{all_scores}\n", file=o)



