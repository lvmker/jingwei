package com.bgi.jingwei.util;


import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.bgi.jingwei.entity.ColorEnum;
import com.bgi.jingwei.entity.exception.BaseException;
/**
 * 图像处理工作类
 * @author yeyuanchun
 *
 */
public class MatProcessTools {

	/**
	 * 从文件地址加载图片
	 * @param imagePath 图片地址
	 * @param flag 加载模式 参见 Imgcodecs.ImreadModes
	 * Imgcodecs.IMREAD_COLOR 1 三通道BGR彩色图像
	 * Imgcodecs.IMREAD_GRAYSCALE 0 单通道灰度图像
	 * Imgcodecs.IMREAD_UNCHANGED -1 按原样返回加载的图像（使用alpha通道，否则将被裁剪）
	 * @return
	 */
	public static Mat initMat(String imagePath,Integer flag) {
		Mat src;
		if(null==flag) {
			src = Imgcodecs.imread(imagePath);
		}else {
			src = Imgcodecs.imread(imagePath, flag);
		}
		if (null==src||src.empty()) {
			return null;
		}
		return src;
	}
	
	/**
	 * 高斯模糊
	 * @param src 原始图像
	 * @param size 高斯内核大小  ksize.width 和ksize.height 可以不同，但是必须同时是正数或者负数
	 * @param sigmaX 高斯内核在X方向的标准差
	 * @param sigmaY 高斯内核在Y方向的标准差
	 * @return
	 */
	public static Mat gaussianBlur(Mat src,Size size,double sigmaX,double sigmaY) {
		Mat dst = new Mat();
		Imgproc.GaussianBlur(src, dst, size, sigmaX, sigmaY, Core.BORDER_DEFAULT);
		if (null==dst||dst.empty()) {
			return null;
		}
		return dst;
	}
	
	/**
	 * 颜色空间转换
	 * @param src 原始图像
	 * @param code 颜色转换编码
	 * @return
	 */
	public static Mat cvtColor(Mat src,Integer code) {
		Mat dst = new Mat();
		Imgproc.cvtColor(src, dst, code);
		if (null==dst||dst.empty()) {
			return null;
		}
		return dst;
	}
	/**
	 * 水平&垂直索贝尔算子融合
	 * @param src 输入图像
	 * @return
	 */
	public static Mat sobelxy(Mat src) {
		Mat sobelx=sobel(src, CvType.CV_16S, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);
		Mat sobely=sobel(src, CvType.CV_16S, 0, 1, 3, 1, 0, Core.BORDER_DEFAULT);
		Mat grad = new Mat();
        Core.addWeighted(sobelx, 1, sobely, 0, 0, grad);//将两个图像融合 alpha 第一个数组权重，beta 第二个数组权重 gamma融合之后添加的数值
        return grad;
	}
	/**
	 * 索贝尔算子
	 * @param src 输入图像
	 * @param ddepth 输出图像深度
	 * @param dx 阶倒数x
	 * @param dy 阶倒数y
	 * @param ksize 扩展Sobel内核的大小；它必须是1、3、5或7
	 * @param scale 计算的导数值的可选比例因子；默认情况下，没有缩放
	 * @param delta 在将结果存储到dst之前添加到结果中的可选增量值。
	 * @param borderType
	 * @return
	 */
	public static Mat sobel(Mat src,Integer ddepth,Integer dx,Integer dy,Integer ksize,Integer scale,Integer delta,Integer borderType) {
		Mat grad_x = new Mat();
		Mat abs_grad_x = new Mat();
		Imgproc.Sobel(src, grad_x, ddepth, dx,dy, ksize, scale, delta,borderType);
		Core.convertScaleAbs(grad_x, abs_grad_x);
		return abs_grad_x;
	}
	
	
	/**
	 * 图像二值化
	 * @param src 输入图像
	 * @param thresh 阈值
	 * @param maxval dst图像中最大值(二值化的大值)
	 * @param type 阈值类型
	 * Imgproc.THRESH_BINARY 超过阈值极大，否则极小
	 * if(src(x,y)>thresh){
	 * dst(x,y)=maxval;
	 * }else{
	 * dst(x,y)=0;
	 * }
	 * Imgproc.THRESH_BINARY_INV 超过阈值极小，否则极大
	 * if(src(x,y)>thresh){
	 * dst(x,y)=0;
	 * }else{
	 * dst(x,y)=maxval;
	 * }
	 * Imgproc.THRESH_TRUNC 超过阈值取阈值，否则取原值
	 * if(src(x,y)>thresh){
	 * dst(x,y)=thresh;
	 * }else{
	 * dst(x,y)=src(x,y);
	 * }
	 * Imgproc.THRESH_TOZERO 超过阈值取原值，否则取极小
	 * if(src(x,y)>thresh){
	 * dst(x,y)=src(x,y);
	 * }else{
	 * dst(x,y)=0;
	 * }
	 * Imgproc.THRESH_TOZERO 超过阈值取极小，否则取原值
	 * if(src(x,y)>thresh){
	 * dst(x,y)=0;
	 * }else{
	 * dst(x,y)=src(x,y);
	 * }
	 * Imgproc.THRESH_OTSU 最大类间方差法  thresh 的值会被忽略，可以任取一个值
	 * @return
	 */
	public static Mat threshold(Mat src,double thresh, double maxval, int type) {
		Mat dst = new Mat();
		Imgproc.threshold(src, dst, thresh, maxval, type);
		return dst;
	}
	/**
	 * 图像膨胀腐蚀操作(使用矩形结构元素)
	 * @param src 输入图像
	 * @param size
	 * @param operate 图像操作
	 * MORPH_ERODE    = 0, //腐蚀
	 * MORPH_DILATE   = 1, //膨胀
	 * MORPH_OPEN     = 2, //开操作 先腐蚀后膨胀
	 * MORPH_CLOSE    = 3, //闭操作 先膨胀后腐蚀
	 * MORPH_GRADIENT = 4, //梯度操作 膨胀图减腐蚀图
	 * MORPH_TOPHAT   = 5, //顶帽操作 原始图像减开操作所得图像
	 * MORPH_BLACKHAT = 6, //黑帽操作 闭操作所得图像减原始图像
	 * MORPH_HITMISS  = 7, //击中击不中  前景背景腐蚀运算的交集。仅仅支持CV_8UC1 二进制图像
	 * @return
	 */
	public static Mat morphologyExRect(Mat src,Size size,int operate) {
		Mat dst = new Mat();
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size);//结构元素
		Imgproc.morphologyEx(src, dst, operate, kernel);
		return dst;
	}
	/**
	 * 判断是否形似车牌的矩形
	 * @param mr
	 * @return
	 */
	public static Mat getPlateMat(Mat src,RotatedRect mr) {
		boolean needAdjust=mr.size.height > mr.size.width;//图形需要调整
		double r =needAdjust?(mr.size.height / mr.size.width):(mr.size.width / mr.size.height);//矩形宽高比,
		double area=mr.size.height * mr.size.width;//矩形面积
		double minArea=44 * 14 * 3;//可以识别的最小车牌面积
		double aspect =3.75;//标准宽高比
	    double rmin = aspect - 1;//
	    double rmax = aspect + 1;//
	    boolean isMatch=area >= minArea&&r >= rmin && r <= rmax;
	    if(isMatch) {
	    	System.out.println("图像面积："+area+"("+minArea+",-)"+",矩形宽高比："+r+"("+rmin+","+rmax+"),ismatch="+isMatch);
		    double angle = mr.angle;
		    Size rect_size = new Size((int) mr.size.width, (int) mr.size.height);
		    if(needAdjust) {
		    	angle = 90 + angle;
		    	rect_size = new Size(rect_size.height, rect_size.width);
		    }
		    if (angle - 30 < 0 && angle + 30 > 0) {
		    	System.out.println("height:"+rect_size.height+",width:"+rect_size.width+",angle:"+angle);
                Mat img_rotated = new Mat();
                Mat rotmat =Imgproc.getRotationMatrix2D(mr.center, angle, 1);
                Imgproc.warpAffine(src, img_rotated, rotmat, src.size());//仿射变换 用户图像平移和旋转
                Mat img_crop = new Mat();
                Imgproc.getRectSubPix(src, rect_size, mr.center, img_crop);
                Mat resultResized = new Mat();
                resultResized.create(36, 136, CvType.CV_8UC3);
                Imgproc.resize(img_crop, resultResized, resultResized.size(), 0, 0, 2);
                return resultResized;
		    }
	    }
		return null;
	}
	/**
	 * 判断是否文字
	 * @param r
	 * @return
	 */
    public static Boolean isCharMat(Mat r) {
        float aspect = 45.0f / 90.0f;
        float charAspect = (float) r.cols() / (float) r.rows();//宽高比
        float error = 0.7f;
        float minHeight = 10f;
        float maxHeight = 35f;
        // We have a different aspect ratio for number 1, and it can be ~0.2
        float minAspect = 0.05f;
        float maxAspect = aspect + aspect * error;
        // area of pixels
        float area = Core.countNonZero(r);//字体面积
        // bb area
        float bbArea = r.cols() * r.rows();//总面积
        // % of pixel in area
        float percPixels = area / bbArea;//字体面积占比
        boolean isMatched=bbArea>=200f&&percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect && r.rows() >= minHeight && r.rows() < maxHeight;
//        System.out.println(j+"=="+isMatched+"==宽高比="+charAspect+",非字体面积="+area+",总面积="+bbArea+",percPixels="+percPixels);
        return isMatched;
    }
    
    /**
     * 进行腐蚀操作
     * @param inMat
     * @return
     */
    public static Mat erode(Mat inMat) {
        Mat result = inMat.clone();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.erode(inMat, result, element);
        return result;
    }

    /**
     * 进行膨胀操作
     * @param inMat
     * @return
     */
    public static Mat dilate(Mat inMat) {
        Mat result = inMat.clone();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.dilate(inMat, result, element);
        return result;
    }
    /**
     * 随机数平移
     * @param inMat
     * @return
     */
    public static Mat randTranslate(Mat inMat) {
        Random rand = new Random();
        Mat result = inMat.clone();
        int ran_x = rand.nextInt(10000) % 5 - 2; // 控制在-2~3个像素范围内
        int ran_y = rand.nextInt(10000) % 5 - 2;
        return translateImg(result, ran_x, ran_y);
    }
    /**
     * 随机数旋转
     * @param inMat
     * @return
     */
    public static Mat randRotate(Mat inMat) {
        Random rand = new Random();
        Mat result = inMat.clone();
        float angle = (float) (rand.nextInt(10000) % 15 - 7); // 旋转角度控制在-7~8°范围内
        return rotateImg(result, angle);
    }


    /**
     * 平移
     * @param img
     * @param offsetx
     * @param offsety
     * @return
     */
    public static Mat translateImg(Mat img, int offsetx, int offsety){
        Mat dst = new Mat();
        //定义平移矩阵
        Mat trans_mat = Mat.zeros(2, 3, CvType.CV_32FC1);
        trans_mat.put(0, 0, 1);
        trans_mat.put(0, 2, offsetx);
        trans_mat.put(1, 1, 1);
        trans_mat.put(1, 2, offsety);
        Imgproc.warpAffine(img, dst, trans_mat, img.size());    // 仿射变换
        return dst;
    }
    
    /**
     * 旋转角度
     * @param source
     * @param angle
     * @return
     */
    public static Mat rotateImg(Mat source, float angle){
        Point src_center = new Point(source.cols() / 2.0F, source.rows() / 2.0F);
        Mat rot_mat = Imgproc.getRotationMatrix2D(src_center, angle, 1);
        Mat dst = new Mat();
        // 仿射变换 可以考虑使用投影变换; 这里使用放射变换进行旋转，对于实际效果来说感觉意义不大，反而会干扰结果预测
        Imgproc.warpAffine(source, dst, rot_mat, source.size());    
        return dst;
    }
    /**
     * 
     * @param in
     * @param sizeData
     * @return
     */
    public static Mat features(Mat in, int sizeData) {
        float[] vhist = projectedHistogram(in, false);
        float[] hhist = projectedHistogram(in, true);
        Mat lowData = new Mat();
        if (sizeData > 0) {
            Imgproc.resize(in, lowData, new Size(sizeData, sizeData));//调整图像大小
        }
        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = new Mat(1, numCols, CvType.CV_32F);
        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            out.put(0, j, vhist[i]);
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            out.put(0, j, hhist[i]);
        }
        for (int x = 0; x < lowData.cols(); x++) {
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                double[] val = lowData.get(x, y);
                out.put(0, j, val[0]);
            }
        }
        return out;
    }
    /**
     * 
     * @param img
     * @param isHorizontal
     * @return
     */
    public static float[] projectedHistogram(final Mat img, boolean isHorizontal) {
        int rows=img.rows();
        int cols=img.cols();
        float[] nonZeroMat;
        if(isHorizontal) {//水平
        	nonZeroMat = new float[rows];
        }else {//垂直
        	nonZeroMat = new float[cols];
        }
        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
        Core.extractChannel(img, img, 0);
        for (int j = 0; j < nonZeroMat.length; j++) {
            Mat data = isHorizontal ? img.row(j) : img.col(j);
            int count = Core.countNonZero(data);
            nonZeroMat[j] = count;
        }
        // Normalize histogram
        float max = 0;
        for (int j = 0; j < nonZeroMat.length; ++j) {
            max = Math.max(max, nonZeroMat[j]);
        }
        if (max > 0) {
            for (int j = 0; j < nonZeroMat.length; ++j) {
                nonZeroMat[j] /= max;
            }
        }
        return nonZeroMat;
    }
    
    /**
     * 统一字符的大小
     * @param in
     * @return
     */
    public static Mat preprocessChar(Mat in) {
        int h = in.rows();
        int w = in.cols();
        Mat transformMat = Mat.eye(2, 3, CvType.CV_32F);
        int m = Math.max(w, h);
        transformMat.put(0, 2, (m - w) / 2f);
        transformMat.put(1, 2, (m - h) / 2f);

        Mat warpImage = new Mat(m, m, in.type());
        Imgproc.warpAffine(in, warpImage, transformMat, warpImage.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));

        Mat resized = new Mat(20, 20, CvType.CV_8UC3);
        Imgproc.resize(warpImage, resized, resized.size(), 0, 0, Imgproc.INTER_CUBIC);

        return resized;
    }
    /**
     * 将Rect按位置从左到右进行排序
     * @param vecRect
     * @param out
     * @return
     */
    public static void sortRect(Vector<Rect> vecRect, Vector<Rect> out) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < vecRect.size(); ++i) {
            map.put(vecRect.get(i).x, i);
        }
        Set<Integer> set = map.keySet();
        Object[] arr = set.toArray();
        Arrays.sort(arr);
        for (Object key : arr) {
            out.add(vecRect.get(map.get(key)));
        }
        return;
    }
	/**
	 * 调整字符图像至可分析（20*20）
	 * @param src
	 * @return
	 */
	public static Mat resizeMat(Mat src) {
        int height = src.rows();
		int width = src.cols();
		boolean isCol=height>width?true:false;//是否补列，原矩形高大于宽，需要增大宽，反之增大高
		int reacSize=isCol?height:width;//新矩形的宽和高
		int offset=Math.abs(height-width)/2;//图像偏移值
		Mat dst=Mat.zeros(reacSize, reacSize, src.type());//目标图像
		for (int row = 0; row < height; row++) {//图像复制
			for (int col = 0; col < width; col++) {
				if(isCol) {
					dst.put(row, col+offset, src.get(row, col));
				}else {
					dst.put(row+offset, col, src.get(row, col));
				}
				
			}
		}
		Imgproc.resize(dst, dst, new Size(20, 20));
		return dst;
	}
	
    /**
     * 获取图片主体颜色
     * @param src
     * @return
     * @throws BaseException
     */
	public static ColorEnum getMatColor(Mat src) throws BaseException {
        Mat hsv = new Mat();
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV);
        ColorEnum dstColor = null;
        int maxNum=-1;
        for(ColorEnum color:ColorEnum.values()) {
            Mat mask = new Mat();
            Core.inRange(hsv, color.lowerb, color.higherb, mask);
    		Mat binaryImageMat = new Mat();
    		Imgproc.threshold(mask, binaryImageMat, 127, 255, Imgproc.THRESH_BINARY );
    		int count=Core.countNonZero(binaryImageMat);
    		System.out.println(color.desc+"=="+count);
    		if(count>maxNum) {
    			dstColor=color;
    			maxNum=count;
    		}
        }
		return dstColor;
	}
}
