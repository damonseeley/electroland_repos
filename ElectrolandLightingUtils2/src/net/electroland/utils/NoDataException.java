package net.electroland.utils;

@SuppressWarnings("serial")
public class NoDataException extends Exception {

	public NoDataException(){
		super();
	}
	public NoDataException(String mssg){
		super(mssg);
	}
}
