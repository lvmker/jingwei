package com.bgi.jingwei.service;

import org.opencv.core.Core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;
/**
 * 项目启动加载方法
 * @author yeyuanchun
 *
 */
@Component
public class ApplicationRunnerImpl implements ApplicationRunner {
	private Logger logger=LoggerFactory.getLogger(ApplicationRunnerImpl.class);
	@Override
	public void run(ApplicationArguments args) throws Exception {
		// TODO Auto-generated method stub
	      String[] sourceArgs = args.getSourceArgs();
	      for (String arg : sourceArgs) {
	    	  logger.info("[系统启动]启动参数-"+arg);
	      }
	      System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	      logger.info("[系统启动]加载opencv动态链接库："+Core.NATIVE_LIBRARY_NAME);
	}

}
