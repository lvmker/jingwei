package com.bgi.jingwei.entity;
/**
 * Mat处理机
 * @author yeyuanchun
 *
 */

import org.opencv.core.Mat;
import org.opencv.core.Size;

import com.bgi.jingwei.entity.exception.BaseException;
import com.bgi.jingwei.util.MatProcessTools;

public class MatProcessor {
	private String imagePath;//图片路径 1
	private Mat sourceMat;//原始图像 2
	private Mat gaussianMat;//高斯模糊图像 3
	public MatProcessor(Mat sourceMat) {
		this.sourceMat=sourceMat;
	}
	public MatProcessor(String imagePath) {
		this.imagePath=imagePath;
		this.sourceMat=MatProcessTools.initMat(imagePath, null);
	}
	/**
	 * 高斯模糊处理
	 * @param size 高斯内核
	 * @param sigmaX 高斯内核在X方向的标准差
	 * @param sigmaY 高斯内核在Y方向的标准差
	 * @throws BaseException 
	 */
	public void gaussianBlur(Size size,double sigmaX,double sigmaY) throws BaseException {
		if(this.sourceMat==null||sourceMat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "sourceMat为空，无法进行高斯模糊");
		}
		this.gaussianMat=MatProcessTools.gaussianBlur(sourceMat, size, sigmaX, sigmaY);
	}
	
}
