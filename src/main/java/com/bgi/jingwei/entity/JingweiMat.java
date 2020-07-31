package com.bgi.jingwei.entity;
/**
 * 精卫Mat
 * @author yeyuanchun
 *
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.bgi.jingwei.entity.exception.BaseException;
import com.bgi.jingwei.util.MatProcessTools;

public class JingweiMat {
	private Mat mat;//图像
	private String desc;//描述
	public JingweiMat(Mat mat,String desc) {
		this.mat=mat;
		this.desc=desc;
	}
	public Mat getMat() {
		return mat;
	}
	public void setMat(Mat mat) {
		this.mat = mat;
	}
	public String getDesc() {
		return desc;
	}
	public void setDesc(String desc) {
		this.desc = desc;
	}

	/**
	 * 高斯模糊处理
	 * @param size 高斯内核
	 * @param sigmaX 高斯内核在X方向的标准差
	 * @param sigmaY 高斯内核在Y方向的标准差
	 * @throws BaseException 
	 */
	public JingweiMat gaussianBlur(Size size,double sigmaX,double sigmaY) throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行高斯模糊");
		}
		Mat dst=MatProcessTools.gaussianBlur(this.mat, size, sigmaX, sigmaY);
		return new JingweiMat(dst,"gaussianblur");
	}
	/**
	 * 图像灰度化（图像颜色空间转换）
	 * @param code 默认为灰度
	 * Imgproc.COLOR_BGR2HSV
	 * @return
	 * @throws BaseException
	 */
	public JingweiMat cvtColor2Gray(Integer code) throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行图像灰度化");
		}
		if(null==code) {
			code=Imgproc.COLOR_BGRA2GRAY;
		}
		Mat dst=MatProcessTools.cvtColor(mat, code);
		return new JingweiMat(dst,"gray");
	}
	/**
	 * 水平垂直索贝尔算子融合
	 * @return
	 * @throws BaseException
	 */
	public JingweiMat sobelxy() throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行水平垂直索贝尔算子融合");
		}
		Mat dst=MatProcessTools.sobelxy(mat);
		return new JingweiMat(dst,"sobelxy");
	}
	/**
	 * 图像二值化
	 * @param thresh
	 * @param maxval
	 * @param type
	 * @return
	 * @throws BaseException
	 */
	public JingweiMat threshold(double thresh, double maxval, int type) throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法生成二值化图像");
		}
		Mat dst=MatProcessTools.threshold(mat, thresh, maxval, type);
		return new JingweiMat(dst,"threshold");
	}
	/**
	 * 矩形结构的图像形态学操作（腐蚀与膨胀）
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
	 * @throws BaseException
	 */
	public JingweiMat morphologyExRect(Size size, int operate) throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行图像形态学操作");
		}
		Mat dst=MatProcessTools.morphologyExRect(mat, size, operate);
		return new JingweiMat(dst,"morphologyExRect");
	}
	
	/**
	 * 图片裁剪
	 * @param colsp
	 * @param rowsp
	 * @return
	 * @throws BaseException
	 */
	public JingweiMat submat(double colsp,double rowsp) throws BaseException {
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行图片裁剪");
		}
		if(0>=colsp||colsp>1||0>=rowsp||rowsp>1) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "colsp和rowsp为比例，必须在(0,1]范围之内");
		}
        int cols = mat.cols();
        int rows = mat.rows();
        Mat dst=mat.submat((int)((1-rowsp)*rows/2),(int) ((1+rowsp)*rows/2), (int)((1-colsp)*cols/2), (int)((1+colsp)*cols/2));
		return new JingweiMat(dst,colsp+"submat"+rowsp);
	}
	
	/**
	 * 轮廓提取
	 * @return
	 * @throws BaseException
	 */
	public Vector<RotatedRect> getRotatedRects() throws BaseException{
		if(this.mat==null||mat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "mat为空，无法进行轮廓提取");
		}
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		Vector<RotatedRect> rects=new Vector<>();
		for (int i = 0; i < contours.size(); i++) {
			RotatedRect mr=Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray()));
			rects.add(mr);
		}
		return rects;
	}

	
	
	
}
