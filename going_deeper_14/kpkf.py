import cv2
import numpy as np

from kalman import KF2d

class tracker:
    def __init__(self, num_points=17, sys_err=0.95, measure_err=100):
        self.l_KF = []
        self.l_P  = []
        self.l_x  = []
        self.num_points = num_points
        for n in range(num_points):
            KF = KF2d(dt=1)
            P = 100 * np.eye(4, dtype=np.float)
            x = np.array([0,0,0,0], dtype=np.float)
            
            self.l_KF.append(KF)
            self.l_P.append(P)
            self.l_x.append(x)
        
        self.l_estimate = []
        self.keypoints = []

    def main_process(self, list_measured_points):
        self.l_measure = list_measured_points
        for i in range(self.num_points):
            point = list_measured_points[i]
            z = np.array(point, dtype=np.float)

            self.l_x[i], self.l_P[i], filtered_point = self.l_KF[i].process(self.l_x[i], self.l_P[i], z)
            
            self.l_estimate.append(filtered_point)

    def preprocess(self, list_measured_points):
        return list_measured_points
    
    def postprocess(self):
        cnt_validpoint = 0
        x_vel_sum, y_vel_sum = 0, 0
        for i in range(self.num_points):
            if self.l_x[i][0] > 10 and self.l_x[i][2] > 10:
                x_vel_sum += abs(self.l_x[i][1])
                y_vel_sum += abs(self.l_x[i][3])
                cnt_validpoint += 1

        x_vel_mean = x_vel_sum / cnt_validpoint if cnt_validpoint != 0 else 0
        y_vel_mean = y_vel_sum / cnt_validpoint if cnt_validpoint != 0 else 0
        
        for i in range(self.num_points):
            if x_vel_mean > 10.0 or y_vel_mean > 10.0:
                self.l_estimate[i] = self.l_measure[i]
                x, y = self.l_measure[i]
                self.l_x[i] = np.array([x,0,y,0], dtype=np.float)

            v = 2 if self.l_estimate[i][0] > 10 and self.l_estimate[i][1] > 10 else 0
            self.keypoints += list(self.l_estimate[i]) + [v]
        
    def process(self, list_measured_points):
        self.keypoints = []
        self.l_estimate = []
        
        #self.preprocess(list_measured_points)
        self.main_process(list_measured_points)
        self.postprocess()
        
        return self.keypoints
