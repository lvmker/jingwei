package com.bgi.jingwei.service;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;
import com.bgi.jingwei.entity.JingweiMat;
import com.bgi.jingwei.entity.exception.BaseException;
import com.bgi.jingwei.service.train.ANNPredict;
import com.bgi.jingwei.util.ImageUI;
import com.bgi.jingwei.util.MatProcessTools;
/**
 * 车牌识别服务
 * @author yeyuanchun
 *
 */
@Service
public class PlateService {
	public void plateRecognition() throws BaseException {
		String imagePath="E:\\4.jpg";
		ANNPredict annPredict=new ANNPredict();
		JingweiMat src=new JingweiMat(MatProcessTools.initMat(imagePath, null), "src");
		new ImageUI().imshow(src);
		JingweiMat grayMat=src.gaussianBlur(new Size(5, 5), 0, 0)
				.cvtColor2Gray(Imgproc.COLOR_BGRA2GRAY);
		JingweiMat thresholdMat=grayMat
				.sobelxy()
				.threshold(0, 255, Imgproc.THRESH_OTSU)
				.morphologyExRect(new Size(17, 3), Imgproc.MORPH_CLOSE);
		new ImageUI().imshow(thresholdMat);
		List<JingweiMat> plateMats=getPlateMats(thresholdMat.getMat(), src.getMat());
		for(JingweiMat plateMat:plateMats) {
//			new ImageUI().imshow(plateMat);
			JingweiMat plateThresholdMat=plateMat
//					.submat(0.8, 0.8)
					.cvtColor2Gray(Imgproc.COLOR_BGRA2GRAY)
					.threshold(0, 255, Imgproc.THRESH_OTSU);
			new ImageUI().imshow(plateThresholdMat);
			List<JingweiMat> charMats=getCharMats(plateThresholdMat.getMat());
			List<String> charStrings=new ArrayList<>();
			for(int i=0;i<charMats.size();i++) {
				if(i==0) {
					String cncodeString=annPredict.predictcn(resizeMat(charMats.get(i).getMat()));
					charStrings.add(annPredict.KEY_CHINESE_MAP.get(cncodeString));
				}else {
					charStrings.add(annPredict.predict(resizeMat(charMats.get(i).getMat())));
				}
			}
			System.out.println(charStrings.toString());
//			for(JingweiMat charMat:charMats) {
//				new ImageUI().imshow(charMat);
//				System.out.println();
//				annPredict.predict(resizeMat(charMat.getMat()));
//			}
//			
		}
		
		
		
		
		
	}
	/**
	 * 调整字符图像至可分析（20*20）
	 * @param src
	 * @return
	 */
	public Mat resizeMat(Mat src) {
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
	 * 获取车牌图片
	 * @param thresholdMat
	 * @param src
	 * @return
	 * @throws BaseException
	 */
	public List<JingweiMat> getPlateMats(Mat thresholdMat,Mat src) throws BaseException{
		if(thresholdMat==null||thresholdMat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "thresholdMat为空，无法进行轮廓提取");
		}
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Imgproc.findContours(thresholdMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		List<JingweiMat> dsts=new ArrayList<>();
		for (int i = 0; i < contours.size(); i++) {
			RotatedRect mr=Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray()));//包覆轮廓的最小斜矩形
			Mat plateMat=MatProcessTools.getPlateMat(src, mr);
			if(null!=plateMat) {
				dsts.add(new JingweiMat(plateMat, "plate"+i));
			}
		}
		return dsts;
	}
	/**
	 * 获取车牌字符图片
	 * @param thresholdMat
	 * @return
	 * @throws BaseException
	 */
	public List<JingweiMat> getCharMats(Mat thresholdMat) throws BaseException {
		if(thresholdMat==null||thresholdMat.empty()) {
			throw new BaseException(BaseException.ERROR_CODE.UNSUPPORTED_METHOD, "thresholdMat为空，无法进行轮廓提取");
		}
		Mat hierarchy = new Mat();
//		thresholdMat.copyTo(hierarchy);
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
  		Imgproc.findContours(thresholdMat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
  		List<JingweiMat> dsts=new ArrayList<>();
  		for (int i = 0; i < contours.size(); i++) {
			Rect mr=Imgproc.boundingRect(contours.get(i));//获取包覆此轮廓的最小矩形
			Mat charMat = new Mat(thresholdMat, mr);
			if(MatProcessTools.isCharMat(charMat)) {
				dsts.add(new JingweiMat(charMat, "char"+i));
			}
		}
  		return dsts;
	}
	
	public static void main(String[] args) throws BaseException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		PlateService plateService=new PlateService();
		plateService.plateRecognition();
	}
}
