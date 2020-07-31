package com.bgi.jingwei.service.train;
import java.util.HashMap;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;

import com.bgi.jingwei.entity.exception.BaseException;
import com.bgi.jingwei.util.ImageUI;
import com.bgi.jingwei.util.MatProcessTools;

public class ANNPredict {

    private ANN_MLP ann = ANN_MLP.create();
    private ANN_MLP anncn = ANN_MLP.create();
    public ANNPredict() {
        ann.clear();
        ann = ANN_MLP.load(DEFAULT_PATH+"anntrain.xml");
        anncn.clear();
        anncn = ANN_MLP.load(DEFAULT_PATH+"anntraincn.xml");
    }
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    public static Map<String, String> KEY_CHINESE_MAP = new HashMap<String, String>();
    static {
        if (KEY_CHINESE_MAP.isEmpty()) {
            KEY_CHINESE_MAP.put("zh_cuan", "川");
            KEY_CHINESE_MAP.put("zh_e", "鄂");
            KEY_CHINESE_MAP.put("zh_gan", "赣");
            KEY_CHINESE_MAP.put("zh_gan1", "甘");
            KEY_CHINESE_MAP.put("zh_gui", "贵");
            KEY_CHINESE_MAP.put("zh_gui1", "桂");
            KEY_CHINESE_MAP.put("zh_hei", "黑");
            KEY_CHINESE_MAP.put("zh_hu", "沪");
            KEY_CHINESE_MAP.put("zh_ji", "冀");
            KEY_CHINESE_MAP.put("zh_jin", "津");
            KEY_CHINESE_MAP.put("zh_jing", "京");
            KEY_CHINESE_MAP.put("zh_jl", "吉");
            KEY_CHINESE_MAP.put("zh_liao", "辽");
            KEY_CHINESE_MAP.put("zh_lu", "鲁");
            KEY_CHINESE_MAP.put("zh_meng", "蒙");
            KEY_CHINESE_MAP.put("zh_min", "闽");
            KEY_CHINESE_MAP.put("zh_ning", "宁");
            KEY_CHINESE_MAP.put("zh_qing", "青");
            KEY_CHINESE_MAP.put("zh_qiong", "琼");
            KEY_CHINESE_MAP.put("zh_shan", "陕");
            KEY_CHINESE_MAP.put("zh_su", "苏");
            KEY_CHINESE_MAP.put("zh_sx", "晋");
            KEY_CHINESE_MAP.put("zh_wan", "皖");
            KEY_CHINESE_MAP.put("zh_xiang", "湘");
            KEY_CHINESE_MAP.put("zh_xin", "新");
            KEY_CHINESE_MAP.put("zh_yu", "豫");
            KEY_CHINESE_MAP.put("zh_yu1", "渝");
            KEY_CHINESE_MAP.put("zh_yue", "粤");
            KEY_CHINESE_MAP.put("zh_yun", "云");
            KEY_CHINESE_MAP.put("zh_zang", "藏");
            KEY_CHINESE_MAP.put("zh_zhe", "浙");
        }
    }

    // 默认的训练操作的根目录E:/workspace/gitee/yx-image-recognition/PlateDetect/train/plate_detect_svm/
    private static final String DEFAULT_PATH = "E:/workspace/gitee/yx-image-recognition/PlateDetect/train/chars_recognise_ann/";
    public String predict(Mat img) {
        Mat f = MatProcessTools.features(img, ANNTrain.predictSize);
        int index = 0;
        double maxVal = -2;
        Mat output = new Mat(1, ANNTrain.strCharacters.length, CvType.CV_32F);
        ann.predict(f, output);  // 预测结果
        for (int j = 0; j < ANNTrain.strCharacters.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }
        // 膨胀
        f = MatProcessTools.features(MatProcessTools.dilate(img), ANNTrain.predictSize);
        ann.predict(f, output);  // 预测结果
        for (int j = 0; j < ANNTrain.strCharacters.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }

        String result = String.valueOf(ANNTrain.strCharacters[index]);
        System.out.println(result);
        return result;
    }
    public String predictcn(Mat img) {
        Mat f = MatProcessTools.features(img, ANNTrain.predictSize);
        int index = 0;
        double maxVal = -2;
        Mat output = new Mat(1, ANNTrain.strCharactersCN.length, CvType.CV_32F);
        anncn.predict(f, output);  // 预测结果
        for (int j = 0; j < ANNTrain.strCharactersCN.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }
        // 膨胀
        f = MatProcessTools.features(MatProcessTools.dilate(img), ANNTrain.predictSize);
        anncn.predict(f, output);  // 预测结果
        for (int j = 0; j < ANNTrain.strCharactersCN.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }

        String result = String.valueOf(ANNTrain.strCharactersCN[index]);
        System.out.println(result);
        return result;
    }
//    public void predict(Mat img) {
//        ann.clear();
//        ann = ANN_MLP.load(MODEL_PATH);
//
//                Mat f = MatProcessTools.features(img, ANNTrain.predictSize);
//
//                int index = 0;
//                double maxVal = -2;
//                Mat output = new Mat(1, ANNTrain.strCharacters.length, CvType.CV_32F);
//                ann.predict(f, output);  // 预测结果
//                for (int j = 0; j < ANNTrain.strCharacters.length; j++) {
//                    double val = output.get(0, j)[0];
//                    if (val > maxVal) {
//                        maxVal = val;
//                        index = j;
//                    }
//                }
//
//                // 膨胀
//                f = MatProcessTools.features(MatProcessTools.dilate(img), ANNTrain.predictSize);
//                ann.predict(f, output);  // 预测结果
//                for (int j = 0; j < ANNTrain.strCharacters.length; j++) {
//                    double val = output.get(0, j)[0];
//                    if (val > maxVal) {
//                        maxVal = val;
//                        index = j;
//                    }
//                }
//
//                String result = String.valueOf(ANNTrain.strCharacters[index]);
//                System.out.println(result);
//
//        return;
//    }

    public static void main(String[] args) throws BaseException {

        ANNPredict annT = new ANNPredict();
//        annT.train(Constant.predictSize, Constant.neurons);
//        String filePath="E:\\workspace\\gitee\\yx-image-recognition\\PlateDetect\\train\\chars_recognise_ann\\learn\\5\\4-4.jpg";
        String filePath="E:\\char.jpg";
        Mat img = Imgcodecs.imread(filePath, 0);
        int height = img.rows();
		int width = img.cols();
        int diff=(int) (img.size().height-img.size().width);
        int half=diff/2;
        int tohalf=diff-half;
        System.out.println(diff+"="+half+"="+tohalf);
        Mat dst=Mat.zeros((int)img.size().height, (int)img.size().height, img.type());
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				dst.put(row, col+half, img.get(row, col));
			}
		}
		new ImageUI().imshow("HSV after deal1",dst);
		Imgproc.resize(dst, dst, new Size(20, 20));
		System.out.println(dst.size().height+"=="+dst.size().width);
//		JingweiMat jingweiMat=new JingweiMat(dst, "src");
//		jingweiMat.submat(20/height, 20/height);
        annT.predict(dst);
        return;
    }


}