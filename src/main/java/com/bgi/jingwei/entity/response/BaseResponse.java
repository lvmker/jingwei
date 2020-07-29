package com.bgi.jingwei.entity.response;
import com.bgi.jingwei.entity.exception.BaseException;

/**
 * 通用返回
 * @author yeyuanchun
 *
 */
public class BaseResponse {

	private String code=BaseException.ERROR_CODE.SUCCESS;//返回码
	private String msg="success";//返回描述
	private Object rows;//返回数据
	private Long total=0l;//总条数
	private Integer pageRows;//每页条数
	private Integer pages;//页码
	public BaseResponse() {
		
	}
	
	public BaseResponse(Object rows) {
		this.rows=rows;
	}
	
	public BaseResponse(String code,String msg) {
		this.code=code;
		this.msg=msg;
	}
	public String getCode() {
		return code;
	}
	public void setCode(String code) {
		this.code = code;
	}
	public String getMsg() {
		return msg;
	}
	public void setMsg(String msg) {
		this.msg = msg;
	}
	public Object getRows() {
		return rows;
	}
	public void setRows(Object rows) {
		this.rows = rows;
	}
	public Long getTotal() {
		return total;
	}
	public void setTotal(Long total) {
		this.total = total;
	}
	public Integer getPageRows() {
		return pageRows;
	}
	public void setPageRows(Integer pageRows) {
		this.pageRows = pageRows;
	}
	public Integer getPages() {
		return pages;
	}
	public void setPages(Integer pages) {
		this.pages = pages;
	}
	@Override
	public String toString() {
		return "BaseResponse [code=" + code + ", msg=" + msg + ", rows=" + rows + ", total=" + total + ", pageRows="
				+ pageRows + ", pages=" + pages + "]";
	}
	
	
}
