package net.electroland.enteractive.screen {
	
	import flash.display.Sprite;
	import flash.events.Event;
	import com.ericfeminella.collections.HashMap;
	import sandy.util.DistortImage;
	
	/*
	LIGHTMATRIX.as
	by Aaron Siegel, 1-20-2010
	
	This contains the displayable Light objects and receives new
	lighting data from the Java application, then updates the
	Light objects with the new state.
	
	*/
	
	public class LightMatrix extends Sprite{
		
		// light variables
		private var lights:HashMap;
		private var horizontalCount:Number = 18;
		private var verticalCount:Number = 6;
		private var lightWidth = 30;
		private var lightHeight = 30;
		private var verticalSpacing = 50;
		private var horizontalSpacing = 12;
		private var lightGrid:Sprite = new Sprite();
		private var distortImage:DistortImage;
		private var distortedGrid:Sprite;
		private var pin1:Sprite;
		private var pin2:Sprite;
		private var pin3:Sprite;
		private var pin4:Sprite;
		
		
		public function LightMatrix(){
			lights = new HashMap();
			
			// add light objects here
			var lightID:Number = 0;
			for(var v:Number = 0; v<verticalCount; v++){
				var ypos:Number = v * (lightHeight + verticalSpacing);
				for(var h:Number = 0; h<horizontalCount; h++){
					var xpos:Number = h * (lightWidth + horizontalSpacing);
					var light:Light = new Light(lightID, xpos, ypos, lightWidth, lightHeight);
					lights.put(lightID, light);
					lightGrid.addChild(light);
					lightID++;
				}
			}
			
			distortedGrid = new Sprite();
			addChild(distortedGrid);
			distortImage = new DistortImage();
			distortImage.target = lightGrid;
			distortImage.container = distortedGrid;
			
			// doesn't need to do this, but good to see the grid when the Enteractive java code is not running.
			distortImage.initialize(8,8);
			distortImage.setTransform(310,209, 1065,86, 1168,663, 216,671);
			distortImage.render();
		}
		
		public function updateLights(xmlData:XML):void{
			// receive lighting data and propogate to light objects
			var lightList:XMLList = xmlData.lights.light;
			for each (var lightElement:XML in lightList){
				//trace(Number(lightElement.toString()));
				var light:Light = lights.getValue(Number(lightElement.attribute("id")));
				light.updateLight(Number(lightElement.toString()));
			}
			
			// must re-initialize, transform, and render distorted grid each update of the light data
			distortImage.initialize(8,8);
			distortImage.setTransform(310,209, 1065,86, 1168,663, 216,671);
			distortImage.render();
		}
		
	}
	
}