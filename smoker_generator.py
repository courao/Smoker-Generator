from mtcnn import MTCNN
import cv2
import numpy as np
import os

class cigar_inserter:
    def __init__(self):
        self.detector = MTCNN()

    def process_cigar(self,fname):
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        x_min = np.min(np.where(alpha > 0)[1])
        x_max = np.max(np.where(alpha > 0)[1])
        y_min = np.min(np.where(alpha > 0)[0])
        y_max = np.max(np.where(alpha > 0)[0])
        pts = np.float32([[x_min, y_min], [x_min, (y_min + y_max) / 2], [x_max, (y_min + y_max) / 2]])
        h = y_max - y_min
        w = x_max - x_min
        return bgr, alpha, pts, h, w

    def detect_mouth(self,fname):
        img = cv2.imread(fname)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.detect_faces(rgb)
        if len(res) > 0:
            key_pts = res[0]['keypoints']
            mouth_left = key_pts['mouth_left']
            mouth_right = key_pts['mouth_right']
            mouth_len = np.linalg.norm([np.array(mouth_left),np.array(mouth_right)])
            print(mouth_len)
            print(mouth_left, mouth_right)
            cv2.circle(img, mouth_left, 2, (255, 128, 30))
            cv2.circle(img, mouth_right, 2, (255, 128, 30))

    def insert_cigar(self,cigar_path,nonsmoking_path,alpha=120):
        img = cv2.imread(nonsmoking_path)
        if img is None:
            return img,[]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.detect_faces(rgb)
        alpha = alpha / 180 * np.pi
        beta = alpha - np.pi / 2
        rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        rot_mat2 = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
        bgr, alp, pts1, h_cigar, w_cigar = self.process_cigar(cigar_path)
        cigar_locs = list()
        if len(res) > 0:
            for r in res:
                key_pts = r['keypoints']
                x,y,h,w = r['box']
                if h*w<400:
                    continue
                mouth_left = np.array(key_pts['mouth_left'])
                mouth_right = np.array(key_pts['mouth_right'])
                # print(mouth_left, mouth_right)
                mouth_len = np.linalg.norm(mouth_left - mouth_right)
                cigar_start = (mouth_left + mouth_right) / 2
                # cigar_start += np.matmul((mouth_right - mouth_left), rot_mat.T)/5
                cigar_end = cigar_start + np.matmul((mouth_right - mouth_left), rot_mat.T)
                cigar_locs.append([cigar_start,cigar_end])
                cigar_another = cigar_start + np.matmul((mouth_right - mouth_left), rot_mat2.T) / w_cigar * h_cigar / 2
                pts2 = np.float32([cigar_another, cigar_start, cigar_end])
                M = cv2.getAffineTransform(pts1, pts2)
                bgr_trans = cv2.warpAffine(bgr, M, (img.shape[1], img.shape[0]))
                alp_trans = cv2.warpAffine(alp, M, (img.shape[1], img.shape[0]))
                img[alp_trans > 250] = bgr_trans[alp_trans > 250]

        return img,cigar_locs

if __name__ == '__main__':
    import sys
    fname = 'p1.jpg'
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        fname = sys.argv[1]
    print('process {}'.format(fname))
    inserter = cigar_inserter()
    img, cigar_locs = inserter.insert_cigar('cigarettes_new/9.png', fname, alpha=60)
    if len(cigar_locs)>0:
        cv2.imwrite(fname+'_processed.jpg',img)
        print('{} processed!'.format(fname))
    else:
        print('{} face not detected!'.format(fname))
