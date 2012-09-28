package net.electroland.gotham.processing;

import java.awt.Dimension;
import java.util.ArrayList;
import net.electroland.elvis.net.GridData;
//import net.electroland.gotham.core.GothamConductor;
import net.electroland.gotham.processing.assets.*;
//import net.electroland.utils.ElectrolandProperties;
import org.apache.log4j.Logger;
import controlP5.ControlEvent;

public class GridOverlay extends GothamPApplet {

	private static final long serialVersionUID = 1L;
	static Logger logger = Logger.getLogger(GothamPApplet.class);
	private Dimension syncArea;
	//private int nStripes; // Num Stripes that begin on screen.

	private int selector = 0; // Which color swatch from the props file to use.

	private long interval = 3000;
	private Timer timer;
	private GridData gd;

	ArrayList<StripeSimple> stripes;

	ColorPalette cp;

	@Override
	public void handle(GridData t) {
		gd = t;
	}

	@Override
	public void setup() {
		System.out.println("hello");
		syncArea = this.getSyncArea();
		colorMode(HSB, 360, 100, 100);


		cp = new ColorPalette(this);
		cp.createNewPalette(0);

	}

	@Override
	public void drawELUContent() {

		int insetLeft = 80;
		int insetTop = 70;
		double dilate = 6.0;
		
		fill(color(0, 0, 50),8); //fill with a light alpha white
		rect(0,0,syncArea.width,syncArea.height); //fill the whole area

		if (gd != null){
			//System.out.println("Grid width: " + gd.width + " height: " + gd.height + " payload: " + gd.data.length);

			int gridXStart = 2;
			int gridXMax = 70;
			int gridYStart = 2; // test start inset on top
			int gridYMax = 25; //all of the height
			
			int hShift = -0;
			
			//int cellWidth = (syncArea.width-(insetLeft*2))/gd.width;
			//int cellHeight = (syncArea.height-(insetTop*2))/gd.height;
			// for insets
			int cellWidth = (syncArea.width-(insetLeft*2))/(gridXMax-gridXStart);
			int cellHeight = (syncArea.height-(insetTop*2))/(gridYMax-gridYStart);
			//logger.info("Cell dims: " + cellWidth + " " + cellHeight);
			//logger.info(gridXMax * cellWidth + " " + gridYMax * cellHeight);


			//for(int y =0; y < gd.height; y++) {
			//for(int x =0; x < gd.width; x++) {
			for(int y = gridYStart; y < gridYMax; y++) {
				for(int x = gridXStart; x < gridXMax; x++) {
					//System.out.print(t.getValue(x, y) + "  ");
					if (gd.getValue(x, y) > 0){
						fill(color(0, 255, gd.getValue(x, y)));
						
						ellipse(syncArea.width-(insetLeft*2)+hShift-y*cellHeight+insetLeft, syncArea.height-(insetTop*2)-x*cellWidth+insetTop, (int)(cellHeight*dilate), (int)(cellWidth*dilate)); //rotated
						//rect(x*cellWidth+insetLeft, y*cellHeight+insetTop, (int)(cellHeight*dilate), (int)(cellWidth*dilate));

					}
				}
				//System.out.println("");
			}
		}



		loadPixels();
		FastBlur.performBlur(pixels, width, height, 30);
		updatePixels();
	}


}