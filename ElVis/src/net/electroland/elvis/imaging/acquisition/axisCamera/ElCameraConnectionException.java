package net.electroland.elvis.imaging.acquisition.axisCamera;

public class ElCameraConnectionException extends Exception {
	String errorMsg;
	
	public ElCameraConnectionException(String errorMsg) {
		this.errorMsg = errorMsg;
	}
	
	public String toString() {
		return "ELCameraException:" + errorMsg;
	}

}
