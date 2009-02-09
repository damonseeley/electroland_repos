package net.electroland.lafm.core;

import org.tanukisoftware.wrapper.WrapperManager;
import org.tanukisoftware.wrapper.WrapperListener;

public class LAFMWrapper implements WrapperListener{
	
	private Conductor conductor;
	
	private LAFMWrapper(){
		
	}

	public Integer start(String[] args) {
		conductor = new Conductor(args);
		return null;
	}

	public int stop(int exitCode) {
		System.out.println("stop event: "+ exitCode);
		return exitCode;
	}
	
	public void controlEvent(int event) {
		System.out.println("wrapper event: "+ event);
		if(WrapperManager.WRAPPER_CTRL_CLOSE_EVENT == event){
			
		}
	}
	
	public static void main(String[] args) {
		WrapperManager.start( new LAFMWrapper(), args );
    }
}
