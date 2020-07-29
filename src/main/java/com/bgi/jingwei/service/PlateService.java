package com.bgi.jingwei.service;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import com.bgi.jingwei.entity.JingweiMat;
import com.bgi.jingwei.entity.exception.BaseException;
import com.bgi.jingwei.util.MatProcessTools;
/**
 * 车牌识别服务
 * @author yeyuanchun
 *
 */
@Service
public class PlateService {
	public void plateRecognition() throws BaseException {
		String imagePath="";
		JingweiMat jingweiMat=new JingweiMat(MatProcessTools.initMat(imagePath, null), "src");
		jingweiMat.gaussianBlur(new Size(5, 5), 0, 0).cvtColor2Gray(Imgproc.COLOR_BGRA2GRAY).sobelxy().threshold(0, 255, Imgproc.THRESH_OTSU).morphologyExRect(new Size(17, 3), Imgproc.MORPH_CLOSE);
	}
	public static void main(String[] args) {
		
	}
}
