package net.electroland.enteractive.screen {
	
	import flash.display.MovieClip;
	
	/*
	ENTERACTIVESCREENMAIN.as
	by Aaron Siegel, 1-20-2010
	
	The document class of the application. A full screen display of
	an image series of the front of the MetLofts building, changing
	images to reflect the real time sunlight conditions. Simulated
	lights are placed over the images and are updated with live lighting
	data from the Java application.
	
	*/
	
	public class EnteractiveDisplayMain extends MovieClip{
		
		var imageSwitcher:ImageSwitcher;
		var xmlServer:XMLServer;
		var lightMatrix:LightMatrix;
		
		public function EnteractiveDisplayMain(){
			imageSwitcher = new ImageSwitcher();
			addChild(imageSwitcher);
			lightMatrix = new LightMatrix();
			addChild(lightMatrix);
			lightMatrix.init();
			xmlServer = new XMLServer(lightMatrix);
		}
		
	}
	
}