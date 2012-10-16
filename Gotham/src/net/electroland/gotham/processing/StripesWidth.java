package net.electroland.gotham.processing;

public class StripesWidth extends FlexingStripes {
//	public StripesWidth(){
//		super();
//	}
	
	
	public void setup(){
		super.setup();
		
		setFlexing(false);
		setPinning(false); //TODO: fix pinning
		setInsert(false);
		affecters.put("WIDTH", "$50$1$2"); //radius, min scaler, max scaler
		
	}	
	public void drawELUContent(){
		super.drawELUContent();
	}
}
