package com.bgi.jingwei.entity.exception;
/**
 * 通用异常
 * @author yeyuanchun
 *
 */
public class BaseException extends Exception{
	/**
	 * 
	 */
	public static BaseException NETERROR_EXCEPTION=new BaseException(ERROR_CODE.NETERROR, "网络异常");
	public static BaseException UNSUPPORTED_METHOD_EXCEPTION=new BaseException(ERROR_CODE.UNSUPPORTED_METHOD, "不支持的方法");
	public static BaseException UNAUTHORIZED_IP_EXCEPTION=new BaseException(ERROR_CODE.UNAUTHORIZED_IP, "IP未授权");
	public static BaseException AUTHORIZATION_FAILED_EXCEPTION=new BaseException(ERROR_CODE.AUTHORIZATION_FAILED, "授权验证失败");
	public static BaseException UNAUTHORIZED_REQUEST_EXCEPTION=new BaseException(ERROR_CODE.UNAUTHORIZED_REQUEST, "请求未授权或无权限访问数据");
	public static BaseException TOKEN_EXPIRED_EXCEPTION=new BaseException(ERROR_CODE.TOKEN_EXPIRED, "用户未登录或者TOKEN过期");
	private static final long serialVersionUID = 1L;
	private String code;
	public BaseException(String code) {
		this(code, null);
	}
	public BaseException(String code, String message) {
		super(message);
		this.code = code;
	}
	public String getCode() {
		return code;
	}
	
	public static interface ERROR_CODE {
		public static final String SUCCESS="200";//请求成功
		/**
		 * 系统相关的异常
		 */
		public static final String UNKNOWN="10000";//未知错误	系统未知错误
		public static final String NETERROR="10001";//网络异常
		public static final String SERVICE_UNAVAILABLE="10002";//服务不可用
		public static final String DATABASE_ERROR="10003";//数据库操作出现异常
		
		/**
		 * 权限相关的异常
		 */
		public static final String CHECKOUT_FAILED="10081";//校验失败
		public static final String UNSUPPORTED_METHOD="10082";//不支持的方法
		public static final String UNAUTHORIZED_IP="10083";//IP未授权
		public static final String AUTHORIZATION_FAILED="10084";//授权验证失败
		public static final String UNAUTHORIZED_REQUEST="10085";//请求未授权或无权限访问数据
		public static final String TOKEN_EXPIRED="10086";//用户未登录或者TOKEN过期
		/**
		 * 参数相关的异常
		 */
		public static final String PARAM_INVALID="11000";//参数错误
		public static final String PARAM_NULL="11001";//必填字段为空
		public static final String PARAM_TOOLONG="11002";//参数过长
		public static final String PARAM_NOEXISTS="11003";//数据项不存在
		public static final String PARAM_ILLEGAL="11004";//JSON对象格式错误
	}
}
