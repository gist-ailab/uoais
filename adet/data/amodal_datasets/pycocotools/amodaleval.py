__author__ = 'yzhu'

import numpy as np
import datetime
import time
from collections import defaultdict
import json
from pycocotools import mask
import copy
import torchfile
import operator
import os.path


class AmodalEval:
    def __init__(self, amodalGt=None, amodalDt=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param amodalGt: amodal object with ground truth annotations
        :param amodalDt: amodal object with detection results(no category)
        :return: None
        '''
        self.amodalGt   = amodalGt              # ground truth amodal API
        self.amodalDt   = amodalDt              # detections amodal API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params()              # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not amodalGt is None:
            self.params.imgIds = sorted(amodalGt.getImgIds())
            self.params.catIds = [1]

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        #
        # def _toMask(objs, coco):
        #     # modify segmentation by reference
        #     for obj in objs:
        #         t = coco.imgs[obj['image_id']]
        #         for region in obj['regions']:
        #             if type(region['segmentation']) == list:
        #                 # format is xyxy, convert to RLE
        #                 region['segmentation'] = mask.frPyObjects([region['segmentation']], t['height'], t['width'])
        #                 if len(region['segmentation']) == 1:
        #                     region['segmentation'] = region['segmentation'][0]
        #                 else:
        #                     # an object can have multiple polygon regions
        #                     # merge them into one RLE mask
        #                     region['segmentation'] = mask.merge(obj['segmentation'])
        #                 if 'area' not in region:
        #                     region['area'] = mask.area([region['segmentation']])[0]
        #             elif type(region['segmentation']) == dict and type(region['segmentation']['counts']) == list:
        #                 region['segmentation'] = mask.frPyObjects([region['segmentation']],t['height'],t['width'])[0]
        #             elif type(region['segmentation']) == dict and \
        #                 type(region['segmentation']['counts'] == unicode or type(region['segmentation']['counts']) == str):
        #                 # format is already RLE, do nothing
        #                 if 'area' not in region:
        #                     region['area'] = mask.area([region['segmentation']])[0]
        #             else:
        #                 raise Exception('segmentation format not supported.')
        p = self.params
        gts=self.amodalGt.loadAnns(self.amodalGt.getAnnIds(imgIds=p.imgIds))
        dts=self.amodalDt.loadAnns(self.amodalDt.getAnnIds(imgIds=p.imgIds))

        if p.useSegm:
            _toMask(dts, self.amodalDt)
            _toMask(gts, self.amodalGt)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], 1].append(gt)
        for dt in dts:
            if 'category_id' not in dt:
                dt['category_id'] = 1
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = []   # per-image per-category evaluation results
        self.queries = []
        self.eval     = {}   # accumulated evaluation results
    
    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        #print 'Running per image evaluation...'
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p
        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]
        computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        for catId in catIds:
            for areaRng in p.areaRng:
                for imgId in p.imgIds:
                    evalRes = evaluateImg(imgId, catId, areaRng, maxDet, p.occRng)
                    self.evalImgs.append(evalRes)
      
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t=%0.2fs).'%(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        # for multiple gt/annotator case, use gt[0] for now.
        dt = dt[0]['regions']; gt = gt[0]['regions']
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.useSegm:
            if p.useAmodalGT:
                g = [g['segmentation'] for g in gt]
            else:
                g = [g['visible_mask'] if 'visible_mask' in g else g['segmentation'] for g in gt]
            
            if p.useAmodalDT:
                d = [d['amodal_mask'] if 'amodal_mask' in d else d['segmentation'] for d in dt]
            else:
                d = [d['segmentation'] for d in dt]
        else:
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        # compute iou between each dt and gt region
        iscrowd = [0 for o in gt]
        ious = mask.iou(d,g,iscrowd)
        return ious

    def exportDtFile(self, fname):
        # save the matched dt, as a field of gt's regions. Then export the file again. 
        if not self.evalImgs:
            print('Please run evaluate() first')
        res = []
        for key, item in self._gts.iteritems():
            gt = item
            while type(gt) == list:
                gt = gt[0]
            res.append(gt)
        with open(fname, 'wb') as output:
            json.dump(res, output)
        return res

    def evaluateImg(self, imgId, catId, aRng, maxDet, oRng):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        #
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        dt = dt[0]['regions']; gt = gt[0]['regions']
        for g in gt:
            if 'ignore' not in g:
                g['ignore'] = 0
            g['_ignore'] = 0 # default
            if p.onlyThings == 1 and g['isStuff'] == 1:
                g['_ignore'] = 1
            if p.onlyThings == 2 and g['isStuff'] == 0:
                g['_ignore'] = 1
            if g['occluded_rate'] < oRng[0] or g['occluded_rate'] > oRng[1]:
                g['_ignore'] = 1

        # sort dt highest score first, sort gt ignore last
        gtind = [ind for (ind, g) in sorted(enumerate(gt), key=lambda ind, g: g['_ignore'])]

        def inv(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse
        inv_gtind = inv(gtind)
        
        gt = [gt[ind] for ind in gtind]
        dt = sorted(dt, key=lambda x: -x['score'])[0:maxDet]
        iscrowd = [0 for o in gt]
        # load computed ious
        N_iou = len(self.ious[imgId, catId])
        ious = self.ious[imgId, catId][0:maxDet, np.array(gtind)] if N_iou >0 else self.ious[imgId, catId]
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        if ious[dind,gind] < iou:
                            continue
                        iou=ious[dind,gind]
                        m=gind
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['order']
                    gtm[tind,m]  = d['id']
        
        gtm = gtm[:,np.array(inv_gtind)]
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        # save matching ids into self._gts
        self._gts[imgId, catId][0]['gtm'] = gtm.tolist()
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['order'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        #print 'Accumulating evaluation results...   '
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        # K0 = len(_pe.catIds)
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list): # length(1)
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list): # length(1)
                Na = a0*I0
                for m, maxDet in enumerate(m_list): # length(4)
                    E = [self.evalImgs[Nk+Na+i] for i in i_list]
                    E = filter(None, E)
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore']  for e in E])
                    npig = len([ig for ig in gtIg if ig == 0])
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        
                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs)
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall':   recall,
        }
        toc = time.time()
        #print 'DONE (t=%0.2fs).'%( toc-tic )

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, verbose=False):
            p = self.params
            iStr        = ' {:<18} {} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}'
            #titleStr    = 'Average Precision' if ap == 1 else 'Average Recall'
            if ap == 1:
                typeStr = '(AP)'
                titleStr = 'Average Precision'
            elif ap == 2:
                typeStr = '(AR)'
                titleStr = 'Average Recall'
            elif ap == 3:
                typeStr = '(Order AR -- ' + p.sortKey + ')'
                titleStr = 'Order Average Recall'
            
            iouStr      = '%0.2f:%0.2f'%(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '%0.2f'%(iouThr)
            areaStr     = areaRng
            maxDetsStr  = '%d'%(maxDets)

            aind = [i for i, aRng in enumerate(['all', 'small', 'medium', 'large']) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # areaRng
                s = s[:,:,:,aind,mind]
            elif ap == 2:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                s = s[:,:,aind,mind]

            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if verbose:
                print(iStr.format(titleStr, typeStr, iouStr, areaStr, maxDetsStr, '%.3f'%(float(mean_s))))
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        maxProp = self.params.maxDets[-1]  
        
        self.stats = np.zeros((12,))
        self.stats[0] = _summarize(1, maxDets = maxProp)
        self.stats[1] = _summarize(1,iouThr=.5, maxDets = maxProp)
        self.stats[2] = _summarize(1,iouThr=.75, maxDets = maxProp)
        self.stats[3] = _summarize(2,maxDets=1)
        self.stats[4] = _summarize(2,maxDets=10)
        self.stats[5] = _summarize(2,maxDets=100)
        if maxProp == 1000:
            self.stats[6] = _summarize(2,maxDets=1000)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95-.5)/.05))+1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00-.0)/.01))+1, endpoint=True)
        self.maxDets = [1,10,100]
        #self.maxDets = [1,10,100,1000]
        self.areaRng = [ [0**2,1e5**2] ]
        self.useSegm = 1
        self.useAmodalGT = 1
        self.onlyThings = 1 # 1: things only; 0: both
        self.useCats = 1
        self.occRng = [0, 1] # occluding ratio filter. not yet support multi filter for now.
