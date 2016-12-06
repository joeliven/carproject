    def hacky_predict(self, X, magic_i1=50, magic_i2=100, magic_j1=50, magic_j2=224-50, magic_thresh_light=.985, magic_thresh_sum=20., verbose=False):
        i_intercept = self.find_intercept(X, display=True)
        X = self.threshold(X, magic_thresh_light)

    def find_intercept(self,X, h=10, w=5, thresh=20, display=False):
        X_red = self.red_threshold(X)
        # check on LEFT:
        intercept = self.find_intercept_loop(X_red, side='R', thresh=thresh)
        if intercept is not None:
            txt = 'i_int: %d' % intercept
        else:
            txt = 'couldnt find'
            intercept = 0
        print 'txt', txt
        if display:
            plt.imshow(X_red)
            plt.text(224, intercept, txt, color='r', fontsize=25, fontweight='bold')
            plt.show()
        if intercept is not None:
            return 223 -intercept
        else:
            return None

    def find_intercept_loop(self, X_red, side, h=10, w=5, thresh=20):
        # check for red line on LEFT or RIGHT side:
        i_intercept = 223 - h
        found_line = False
        while found_line == False:
            if i_intercept < 0:
                 break
            if side == 'L':
                line_sum = X_red[i_intercept:i_intercept+h,0:w].sum()
            else:
                line_sum = X_red[i_intercept:i_intercept+h,-w:].sum()
            print 'i_intercept', i_intercept+ (h//2), 'line_sum', line_sum
            if line_sum >= thresh:
                found_line = True
            i_intercept -= 1
        i_intercept += (h//2)
        if found_line == False:
            return None
        else:
            return i_intercept

    def red_threshold(self, X, b_thresh=.1, g_thresh=.1, r_thresh=.5):
        b = X[:,:,0]
        g = X[:,:,1]
        r = X[:,:,2]
        bm = np.where(b < b_thresh, 1., 0.)
        gm = np.where(g < g_thresh, 1., 0.)
        rm = np.where(r >= r_thresh, 1., 0.)
        mask = np.asarray([bm, gm, rm]).transpose(1, 2, 0)
        X_red = X * mask
        return X_red

