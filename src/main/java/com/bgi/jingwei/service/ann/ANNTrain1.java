package com.bgi.jingwei.service.ann;
import java.util.Arrays;
import java.util.Vector;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import org.opencv.ml.TrainData;

import com.bgi.jingwei.service.train.ANNTrain;
import com.bgi.jingwei.util.FileUtil;
import com.bgi.jingwei.util.MatProcessTools;
/**
 * 基于org.opencv包实现的训练
 * 
 * 图片文字识别训练
 * 训练出来的库文件，用于识别图片中的数字及字母
 * 
 * 测试了一段时间之后，发现把中文独立出来识别，准确率更高一点
 * 
 * 训练的ann.xml应用：
 * 1、替换res/model/ann.xml文件
 * 2、修改com.yuxue.easypr.core.CharsIdentify.charsIdentify(Mat, Boolean, Boolean)方法
 * 
 * @author yuxue
 * @date 2020-05-14 22:16
 */
public class ANNTrain1 {

    private ANN_MLP ann = ANN_MLP.create();

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    public final static String strCharacters_default[] = {
    		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
            "A", "B", "C", "D", "E", "F", "G", "H", /* 没有I */ "J", "K", "L", "M", "N", /* 没有O */"P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
            ,
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
    // 默认的训练操作的根目录E:/workspace/gitee/yx-image-recognition/PlateDetect/train/plate_detect_svm/
    private static final String DEFAULT_PATH = "E:/workspace/gitee/yx-image-recognition/PlateDetect/train/chars_recognise_ann/";
    private static String ANN_TRAIN_BASE_FOLDER="E:/workspace/gitee/yx-image-recognition/PlateDetect/train/chars_recognise_ann/";//

    // 训练模型文件保存位置
    private static final String MODEL_PATH = DEFAULT_PATH + "anntrain.xml";
    public void train(int in) {
    	String strCharacters[] =Arrays.copyOf(strCharacters_default, in);
    	Mat samples = new Mat(); // 使用push_back，行数列数不能赋初始值
    	Vector<Integer> trainingLabels = new Vector<Integer>();
    	for(int i = 0; i < strCharacters.length; i++) {
    		String charFolder = ANN_TRAIN_BASE_FOLDER + "learn/" + strCharacters[i];//字符样本文件夹
    		System.out.println("start learn "+strCharacters[i]);
            Vector<String> filenames = new Vector<String>();
            FileUtil.getFiles(charFolder, filenames);  // 获取样本文件夹下的所有样本   文件名不能包含中文
            if(!filenames.isEmpty()) {
            	for(String filename:filenames) {//样本数量由样本库决定
            		Mat img = Imgcodecs.imread(filename, 0);
            		/*
            		 * 原始样本
            		 */
            		samples.push_back(MatProcessTools.features(img, ANNTrain.predictSize));//往samples追加数据
            		trainingLabels.add(i);// 每一幅字符图片所对应的字符类别索引下标
            		/*
            		 * 随机平移样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.randTranslate(img), ANNTrain.predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 随机旋转样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.randRotate(img), ANNTrain.predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 膨胀样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.dilate(img), ANNTrain.predictSize));//
            		trainingLabels.add(i);// 
            		/*
            		 * 膨胀样本
            		 */
            		samples.push_back(MatProcessTools.features(MatProcessTools.erode(img), ANNTrain.predictSize));//
            		trainingLabels.add(i);// 
            		
            	}
            }
            System.out.println("done");
    	}
    	samples.convertTo(samples, CvType.CV_32F);
    	Mat classes = Mat.zeros(trainingLabels.size(), strCharacters.length, CvType.CV_32F);
        for (int i = 0; i < trainingLabels.size(); ++i) {
            classes.put(i, trainingLabels.get(i), 1.f);
        }
        TrainData train_data = TrainData.create(samples, Ml.ROW_SAMPLE, classes);
        Mat layers = new Mat(1, 3, CvType.CV_32F);
        layers.put(0, 0, samples.cols());   // 样本特征数 140  10*10 + 20+20
        layers.put(0, 1, ANNTrain.neurons); // 神经元个数
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
        ann.save(ANN_TRAIN_BASE_FOLDER+ "anntrain.xml");
        System.out.println("train"+in+" finish");
    }
    
    public void predict(int in) {
    	String strCharacters[] =Arrays.copyOf(strCharacters_default, in);
        ann.clear();
        ann = ANN_MLP.load(MODEL_PATH);
//
        int total = 0;
        int correct = 0;

        // 遍历测试样本下的所有文件，计算预测准确率
        for (int i = 0; i < strCharacters.length; i++) {

            String c = strCharacters[i];
            String path = DEFAULT_PATH + "learn/" + c;

            Vector<String> files = new Vector<String>();
            FileUtil.getFiles(path, files);

            for (String filePath : files) {

                Mat img = Imgcodecs.imread(filePath, 0);
                Mat f = MatProcessTools.features(img, 10);

                int index = 0;
                double maxVal = -2;
                Mat output = new Mat(1, strCharacters.length, CvType.CV_32F);
                ann.predict(f, output);  // 预测结果
                for (int j = 0; j < strCharacters.length; j++) {
                    double val = output.get(0, j)[0];
                    if (val > maxVal) {
                        maxVal = val;
                        index = j;
                    }
                }

                // 膨胀
                f = MatProcessTools.features(MatProcessTools.dilate(img), 10);
                ann.predict(f, output);  // 预测结果
                for (int j = 0; j < strCharacters.length; j++) {
                    double val = output.get(0, j)[0];
                    if (val > maxVal) {
                        maxVal = val;
                        index = j;
                    }
                }

                String result = String.valueOf(strCharacters[index]);
//                System.out.println(result);
                if(result.equals(String.valueOf(c))) {
                    correct++;
                } else {
                    // 删除异常样本
                    /*File f1 = new File(filePath);
                    f1.delete();*/

//                    System.err.print(filePath);
//                    System.err.println("\t预测结果：" + result);
                }
                total++;
            }

        }

//        System.out.print("total:" + total);
//        System.out.print("\tcorrect:" + correct);
//        System.out.print("\terror:" + (total - correct));
        System.out.println(in+"\t计算准确率为：" + correct / (total * 1.0));

        //牛逼，我操     total:13178  correct:13139   error:39    计算准确率为：0.9970405220822584

        return;
    }

    public static void main(String[] args) {

    	System.out.println(ANNTrain1.strCharacters_default.length);
//    	
//    	for(int i=36;i<ANNTrain1.strCharacters_default.length;i++) {
    		ANNTrain1 annT = new ANNTrain1();
    		annT.train(45);
    		annT.predict(45);
//    	}
//        
        return;
    }


}