package com.bgi.jingwei.service.train;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;
import com.bgi.jingwei.util.FileUtil;
import com.bgi.jingwei.util.MatProcessTools;

public class ANNTrain {
    public static int predictSize = 10;
    public static int neurons = 40;
    public final static String strCharacters[] = {
    		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", /* 没有I */ "J", "K", "L", "M", "N", /* 没有O */"P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
//            ,
//            "zh_cuan",  /*川*/
//            "zh_e",     /*鄂*/
//            "zh_gan",   /*赣*/
//            "zh_gan1",  /*甘*/
//            "zh_gui",   /*贵*/
//            "zh_gui1",  /*桂*/
//            "zh_hei",   /*黑*/
//            "zh_hu",    /*沪*/
//            "zh_ji",    /*冀*/
//            "zh_jin",   /*津*/
//            "zh_jing",  /*京*/
//            "zh_jl",    /*吉*/
//            "zh_liao",  /*辽*/
//            "zh_lu",    /*鲁*/
//            "zh_meng",  /*蒙*/
//            "zh_min",   /*闽*/
//            "zh_ning",  /*宁*/
//            "zh_qing",  /*青*/
//            "zh_qiong", /*琼*/
//            "zh_shan",  /*陕*/
//            "zh_su",    /*苏*/
//            "zh_sx",    /*晋*/
//            "zh_wan",   /*皖*/
//            "zh_xiang", /*湘*/
//            "zh_xin",   /*新*/
//            "zh_yu",    /*豫*/
//            "zh_yu1",   /*渝*/
//            "zh_yue",   /*粤*/
//            "zh_yun",   /*云*/
//            "zh_zang",  /*藏*/
//            "zh_zhe"    /*浙*/        
    };
    
    public final static String strCharactersCN[] = {
            "zh_cuan",  /*川*/
            "zh_e",     /*鄂*/
            "zh_gan",   /*赣*/
            "zh_gan1",  /*甘*/
            "zh_gui",   /*贵*/
            "zh_gui1",  /*桂*/
            "zh_hei",   /*黑*/
            "zh_hu",    /*沪*/
            "zh_ji",    /*冀*/
            "zh_jin",   /*津*/
            "zh_jing",  /*京*/
            "zh_jl",    /*吉*/
            "zh_liao",  /*辽*/
            "zh_lu",    /*鲁*/
            "zh_meng",  /*蒙*/
            "zh_min",   /*闽*/
            "zh_ning",  /*宁*/
            "zh_qing",  /*青*/
            "zh_qiong", /*琼*/
            "zh_shan",  /*陕*/
            "zh_su",    /*苏*/
            "zh_sx",    /*晋*/
            "zh_wan",   /*皖*/
            "zh_xiang", /*湘*/
            "zh_xin",   /*新*/
            "zh_yu",    /*豫*/
            "zh_yu1",   /*渝*/
            "zh_yue",   /*粤*/
            "zh_yun",   /*云*/
            "zh_zang",  /*藏*/
            "zh_zhe"    /*浙*/        
    };
    private ANN_MLP ann = ANN_MLP.create();
    private static String ANN_TRAIN_BASE_FOLDER="E:/workspace/gitee/yx-image-recognition/PlateDetect/train/chars_recognise_ann/";//
    public void train(String strCharacters[],String fileName) {
    	Mat samples = new Mat(); // 使用push_back，行数列数不能赋初始值
    	Vector<Integer> trainingLabels = new Vector<Integer>();
    	for(int i = 0; i < strCharacters.length; i++) {
    		System.out.println("start learn "+strCharacters[i]);
    		String charFolder = ANN_TRAIN_BASE_FOLDER + "learn/" + strCharacters[i];//字符样本文件夹
            Vector<String> filenames = new Vector<String>();
            FileUtil.getFiles(charFolder, filenames);  // 获取样本文件夹下的所有样本   文件名不能包含中文
            if(!filenames.isEmpty()) {
            	for(String filename:filenames) {//样本数量由样本库决定
            		Mat img = Imgcodecs.imread(filename, 0);
            		/*
            		 * 原始样本
            		 */
            		samples.push_back(MatProcessTools.features(img, predictSize));//往samples追加数据
            		trainingLabels.add(i);// 每一幅字符图片所对应的字符类别索引下标
            		/*
            		 * 随机平移样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.randTranslate(img), predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 随机旋转样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.randRotate(img), predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 膨胀样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.dilate(img), predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 膨胀样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.erode(img), predictSize));//
            		trainingLabels.add(i);// 
            		
            	}
            }
            
    	}
    	samples.convertTo(samples, CvType.CV_32F);
    	Mat classes = Mat.zeros(trainingLabels.size(), strCharacters.length, CvType.CV_32F);
        for (int i = 0; i < trainingLabels.size(); ++i) {
            classes.put(i, trainingLabels.get(i), 1.f);
        }
        System.out.println("samples.convertTo(samples, CvType.CV_32F)");
        TrainData train_data = TrainData.create(samples, Ml.ROW_SAMPLE, classes);
        Mat layers = new Mat(1, 3, CvType.CV_32F);
        layers.put(0, 0, samples.cols());   // 样本特征数 140  10*10 + 20+20
        layers.put(0, 1, neurons); // 神经元个数
        layers.put(0, 2, classes.cols());   // 字符数
        ann.clear();
        ann.setLayerSizes(layers);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30000, 0.0001);
        //终止条件
        ann.setTermCriteria(criteria);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.train(train_data);
        ann.save(ANN_TRAIN_BASE_FOLDER+ fileName);
    }
    
    

    
    public static void main(String[] args) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		ANNTrain annTrain=new ANNTrain();
		annTrain.train(ANNTrain.strCharacters,"anntrain.xml");
//		annTrain.train(ANNTrain.strCharactersCN,"anntraincn.xml");
	}
}
