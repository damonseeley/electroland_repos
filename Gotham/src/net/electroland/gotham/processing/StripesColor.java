package net.electroland.gotham.processing;

public class StripesColor extends FlexingStripes {

	public void setup(){
		super.setup();
		
		setFlexing(false);
		setPinning(false); //TODO: fix pinning
		setInsert(false);
		affecters.put("SATURATION", "$190$50$100"); // radius, min, max
		affecters.put("HUE", "$150"); // radius
		
	}
	
	public void drawELUContent(){
		super.drawELUContent();
	}
}
