package com.bgi.jingwei.util;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
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
	public static boolean isPlateRect(RotatedRect mr) {
		boolean needAdjust=mr.size.width > mr.size.height;//图形需要调整
		double r =needAdjust?(mr.size.width / mr.size.height):(mr.size.height / mr.size.width);//矩形宽高比,
		double area=mr.size.height * mr.size.width;//矩形面积
		double minArea=44 * 14 * 3;//可以识别的最小车牌面积
		double aspect =3.75;//标准宽高比
	    double rmin = aspect - 1;//
	    double rmax = aspect + 1;//
	    boolean isMatch=area >= minArea&&r >= rmin && r <= rmax;
	    double angle = mr.angle;
	    if(needAdjust) {
	    	angle = 90 + angle;
	    }
	    
		return isMatch;
	}
	
	
}
