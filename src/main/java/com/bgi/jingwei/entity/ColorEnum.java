package com.bgi.jingwei.entity;

import org.opencv.core.Scalar;

public enum ColorEnum {
	BLACK("black", "黑色",new Scalar(0, 0, 0), new Scalar(180, 255, 46)),
	GRAY("gray", "灰色", new Scalar(0, 0, 46), new Scalar(180, 43, 220)),
	WHITE("white", "白色", new Scalar(0, 0, 221), new Scalar(180, 30, 255)),
	RED("red", "红色", new Scalar(156, 43, 46), new Scalar(180, 255, 255)),
	RED2("red2", "红色2", new Scalar(0, 43, 46), new Scalar(10, 255, 255)),
	ORANGE("orange", "橙色", new Scalar(11, 43, 46), new Scalar(25, 255, 255)),
	YELLOW("yellow", "黄色", new Scalar(26, 43, 46), new Scalar(34, 255, 255)),
	GREEN("green", "绿色", new Scalar(35, 43, 46), new Scalar(77, 255, 255)),
	CYAN("cyan", "青色", new Scalar(78, 43, 46), new Scalar(99, 255, 255)),
	BLUE("blue", "蓝色", new Scalar(100, 43, 46), new Scalar(124, 255, 255)),
	PURPLE("purple", "紫色", new Scalar(125, 43, 46), new Scalar(155, 255, 255));
    public String color;
    public String desc;
    public Scalar lowerb;//
    public Scalar higherb;//
    ColorEnum(String color,String desc,Scalar lowerb,Scalar higherb) {
    	this.color=color;
    	this.desc=desc;
    	this.lowerb=lowerb;
    	this.higherb=higherb;
    }
	
}
