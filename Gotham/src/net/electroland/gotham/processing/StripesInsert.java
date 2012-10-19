package net.electroland.gotham.processing;

@SuppressWarnings("serial")
public class StripesInsert extends FlexingStripes {
	public StripesInsert(){
		super();
	}
	
	public void setup(){
		setFlexing(false);
		setPinning(false); //TODO: fix pinning
		setInsert(true);		
	}	
}
