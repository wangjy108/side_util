import cv2
import math
import numpy as np
import os
import sys
import subprocess
from rdkit import Chem
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import argparse

from img2txt import img2txt
import logging

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

class Img4process():
    def __init__(self, path, N_in_line, N_in_column, if_header, **args):
        self._path = path
        self.N_in_line = N_in_line
        self.N_in_column = N_in_column
        self.if_header = if_header

        try:
            self.auto_terval = args["auto_terval"]
        except Exception as e:
            self.auto_terval = 0.1
        
        #try:
        #    self._path = args["path"]
        #except Exception as e:
        #    self._path = None
        
        
    def region_define(self, gray_array, demension_fac, axis:int):
        #_dic_shape_axis = {"vertical": 0, "horizonal": 1}
        #_dic_ndarray_axis = {"vertical": 1, "horizonal": 0}
        if len(gray_array.shape) > 2:
            gray_array = cv2.cvtColor(gray_array, cv2.COLOR_BGR2GRAY)
        
        _func = lambda x: 1 if x < 255 * int(gray_array.shape[axis]/float(demension_fac)) else (0)
        _get = np.array(list(map(_func, np.sum(gray_array, axis=axis))))
        
        i = 1
        collect_idx = []
        while i < _get.shape[0] - 1:
            a = _get[i]
            a_pre = _get[i-1]
            a_pos = _get[i+1]

            if a * a_pre != 0 and a * a_pos == 0 and a != 0:
                collect_idx.append(i)

            i += 1
        
        return collect_idx
    
    def auto_region_define(self, gray_array, N_in_line, N_in_column, terval, if_header):
        ##N_in_line, axis=1, vertical
        ##N_in_column, axis=0, horizonal
        i = 2
        while i > 0:
            if if_header:
                collect_vertical = self.region_define(gray_array, i, 1)[1:]
            else:
                collect_vertical = self.region_define(gray_array, i, 1)
            if len(collect_vertical) == N_in_line +1:
                break
            else:
                i -= terval
            collect_vertical = []
        
        j = 2
        while j > 0:
            collect_horizonal = self.region_define(gray_array, j, 0)
            if len(collect_horizonal) == 2 * N_in_column +1:
                break
            else:
                j -= terval
            collect_horizonal = []
        
        _dic = {"vertical": collect_vertical,
                "horizonal": collect_horizonal,
                "vertical_dimension_fac": i,
                "horizonal_dimension_fac": j}
        
        return _dic

    def shrink(self, fig_array):
        _array = cv2.cvtColor(fig_array, cv2.COLOR_BGR2GRAY)
        ## vertical_upper
        _func = lambda x: 1 if (x > 255 * _array.shape[1] * 0.7) else -1 if x < 255 * _array.shape[1] * 0.3 else 0
        _get = np.array(list(map(_func, np.sum(_array, axis=1))))


        _dic = {}
        i = 1
        while i < _get.shape[0] - 1:
            a = _get[i]
            a_pre = _get[i-1]
            a_pos = _get[i+1]

            if a * a_pre < 0 and a * a_pos == 1 and a > 0:
                _dic.setdefault("v_start", i)
            elif a * a_pre == 1 and a * a_pos <= 0 and a > 0:
                _dic.setdefault("v_end", i)

            i += 1

        _func2 = lambda x: 1 if (x > 255 * _array.shape[0] * 0.7) else -1 if x < 255 * _array.shape[0] * 0.3 else 0
        _get2 = np.array(list(map(_func2, np.sum(_array, axis=0))))

        j = 1
        while j < _get2.shape[0] - 1:
            a = _get2[j]
            a_pre = _get2[j-1]
            a_pos = _get2[j+1]

            if a * a_pre < 0 and a * a_pos == 1 and a > 0:
                _dic.setdefault("h_start", j)
            elif a * a_pre == 1 and a * a_pos <= 0 and a > 0:
                _dic.setdefault("h_end", j)

            j += 1

        return _dic
    
    def segment(self, dic, gray):
        try:
            vertical_idx = dic["vertical"]
            horizonal_idx = dic["horizonal"]
        except Exception as e:
            return None
        else:
            if not (vertical_idx and horizonal_idx):
                return None

        vertical_pair = [(vertical_idx[i], vertical_idx[i+1]) for i in range(len(vertical_idx)-1)]
        horizonal_pair = [(horizonal_idx[i], horizonal_idx[i+1]) for i in range(len(horizonal_idx)-1)]
        
        for ii, vv in enumerate(vertical_pair):
            for jj, hh in enumerate(horizonal_pair):
                if jj % 2 == 1:
                    type_prefix = "Mol"
                else:
                    type_prefix = "Label"
                mol_prefix = jj // 2
                if not os.path.exists(f"{ii}_{mol_prefix}"):
                    os.mkdir(f"{ii}_{mol_prefix}")
                save_path = os.path.join(os.getcwd(),f"{ii}_{mol_prefix}")
                abs_save_file_name = os.path.join(save_path,f"{type_prefix}_{ii}_{mol_prefix}.png")
                img_array = gray[vv[0]:vv[1], hh[0]:hh[1], :]
                _dic = self.shrink(img_array)
                if len(_dic.keys()) == 4:
                    save_array = img_array[_dic["v_start"]:_dic["v_end"], _dic["h_start"]:_dic["h_end"], :]
                else:
                    save_array = img_array
    
                cv2.imwrite(abs_save_file_name, save_array)
    
    def run(self):
        work_dir = os.getcwd()
        gray = cv2.imread(self.fig)
        prefix = self.fig.split(".")[0]
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        os.chdir(prefix)
        gray_array = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _dic = self.auto_region_define(gray_array, self.N_in_line, self.N_in_column, self.auto_terval, self.if_header)
        self.segment(_dic, gray)
        os.chdir(work_dir)
    
    def run_serial(self):
        main_dir = os.getcwd()
        try:
            os.chdir(self._path)
        except Exception as e:
            logging("no/wrong path specified, check and run again")
            return 
        img_list = [cc for cc in os.listdir("./") if "." in cc]
        sorted_img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))

        ## start_define
        auto_define_1st = sorted_img_list[0]
        gray = cv2.imread(auto_define_1st)
        prefix = auto_define_1st.split(".")[0]
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        os.chdir(prefix)
        gray_array = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _dic = self.auto_region_define(gray_array, 
                                       self.N_in_line, 
                                       self.N_in_column, 
                                       self.auto_terval, 
                                       self.if_header)
        self.segment(_dic, gray)
        os.chdir("../")

        try:
            vertical_dimension_fac = _dic["vertical_dimension_fac"]
            horizonal_dimension_fac = _dic["horizonal_dimension_fac"]
        except Exception as e:
            logging.info("failed with auto serial mode, run with seperate setting")
            return 
        
        for idx, res in enumerate(sorted_img_list[1:]):
            _gray = cv2.imread(res)
            _prefix = res.split(".")[0]
            if not os.path.exists(_prefix):
                os.mkdir(_prefix)
            os.chdir(_prefix)
            _gray_array = cv2.cvtColor(_gray, cv2.COLOR_BGR2GRAY)

            if not (vertical_dimension_fac and horizonal_dimension_fac):
                logging.info("failed with auto serial mode, run with seperate setting")
                return 
            
            ## define_verical
            try:
                _vertical = self.region_define(_gray_array, vertical_dimension_fac, 1)
                _horizonal = self.region_define(_gray_array, horizonal_dimension_fac, 0)
            except Exception as e:
                logging.info("failed with auto serial mode, run with seperate setting")
                return
            if args.if_header:
                _vertical = _vertical[1:]

            _dic_each = {"vertical": _vertical,
                         "horizonal": _horizonal}
            self.segment(_dic_each, _gray)
            os.chdir("../")
        
        os.chdir(main_dir)

class proceed():
    def __init__(self, path):
        self.path = path
        self.abs_path = self.path.split("/")[-1]
        self.main_dir = os.getcwd()

        os.chdir(self.path)
        try:
            self.mol_input = [ff for ff in os.listdir(".") if ff.startswith("Mol")][0]
            self.label_input = [ff for ff in os.listdir(".") if ff.startswith("Label")][0]
        except Exception as e:
            self.mol_input = None
            self.label_input = None
        

    def get_mol(self):
        if not self.mol_input:
            return None
        _dic = {}
        cmd = ["#!/bin/sh \n",\
               "\n",
               "source activate\n",
               "conda activate patentrec\n",
               "python <<_EOF \n",
               "from Image2mol import Img2mol\n",
               f"Img2mol('{self.mol_input}').run() \n"
               "_EOF\n",
               "conda deactivate\n"
               "\n"]
        with open("in.sh", "w+") as cc:
            for line in cmd:
                cc.write(line)
        
        (_status, _out) = subprocess.getstatusoutput("bash in.sh")
        #(_, _) = subprocess.getstatusoutput("conda deactivate")

        try:
            mol = [cc for cc in Chem.SDMolSupplier(f"{self.mol_input.split('.')[0]}.sdf", removeHs=False) if cc]
        except Exception as e:
            _dic.setdefault(self.abs_path, None)
            pass
        else:
            if not mol:
                _dic.setdefault(self.abs_path, None)
            else:
                _dic.setdefault(self.abs_path, mol[0])
        
        return _dic
    
    def get_label(self):
        if not self.label_input:
            return None
        _dic = {}
        content = img2txt(self.label_input).run()
        if content:
            _dic.setdefault(self.abs_path, content[0])
        else:
            _dic.setdefault(self.abs_path, '')
        return _dic
    
    def assemble(self):
        dic_mol = self.get_mol()
        dic_label = self.get_label()

        if not (dic_mol and dic_label):
            os.chdir(self.main_dir)
            os.system(f"rm -rf {self.path}")
            return None

        if dic_mol[self.abs_path]:
            try:
                int(dic_label[self.abs_path])
            except Exception as e:
                real_label = ""
            else:
                real_label = dic_label[self.abs_path]
            
            dic_mol[self.abs_path].SetProp("_Name", real_label)
            cc = Chem.SDWriter(os.path.join(self.main_dir, f"{self.abs_path}_{real_label}.sdf"))
            cc.write(dic_mol[self.abs_path])
            cc.close()
            #logging.info(f"Sucessful to get {dic_label[self.abs_path]} in this page")

        os.chdir(self.main_dir)
        os.system(f"rm -rf {self.path}")

def run_proceed(path_list):
    for idx, path in enumerate(path_list):
        proceed(path).assemble()

if __name__ == "__main__":
    main_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='patent mol rec')
    parser.add_argument('--Input', type=str, required=True, 
                        help='input patent root path')
    parser.add_argument('--N_in_line', type=int, required=True, 
                        help='number of mol in horizonal axis in 1st page of patent')
    parser.add_argument('--N_in_column', type=int, required=True, 
                        help='number of mol in vertical axis in 1st page of patent')
    parser.add_argument('--if_header', type=bool, default=False, 
                        help='if header in input table')
    parser.add_argument('--parallel', type=bool, default=False, 
                        help='if use parallel run, default False')

    args = parser.parse_args()

    Img4process(args.Input, args.N_in_line, args.N_in_column, args.if_header).run_serial()
    #Img4process(args.Input, args.if_header).run_serial()
    
    #print (args.InputFig.split(".")[0])
    logging.info(f"Processing {args.Input}")
    os.chdir(args.Input)
    _in_need_fig = [cc.split(".")[0] for cc in os.listdir("./") if "." in cc]

    for idx, each_fig in enumerate(_in_need_fig):
        try:
            os.path.exists(each_fig)
        except Exception as e:
            logging.info(f"Failed to redirect to {args.Input}: path is not exisit")
            pass
        else:
            os.chdir(each_fig)
            _path = [cc for cc in os.listdir(".")]
            #print(_path)
            if not _path:
                logging.info(f"Faied with {each_fig}, check and run again")
                os.chdir("../")
                os.system(f"rm -f {each_fig}")
            else:
                if args.parallel:
                    n_thread = max(int(cpu_count() / 2), 1)
                    if n_thread > 1:
                        while True:
                            n_in_thread = math.ceil(len(_path) / n_thread)
                            if math.ceil(len(_path) / n_in_thread) == n_thread:
                                break
                            
                            n_thread -= 1
                    else:
                        n_in_thread = len(_path)
                    Parallel(n_jobs=n_thread)(\
                                            delayed(run_proceed)(\
                                            _path[i*n_in_thread:(i+1)*n_in_thread]) for i in range(n_thread))
                else:
                    run_proceed(_path)
                os.chdir("../")
    os.chdir(main_dir)
