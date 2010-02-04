package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	
	public class ControlPanel extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		private var bg:MovieClip;
		
		private var attractionRadiusMaxSlider:ScrollBar;
		private var attractionRadiusMinSlider:ScrollBar;
		private var repulsionRadiusSlider:ScrollBar;
		private var massSlider:ScrollBar;
		private var torqueSlider:ScrollBar;
		private var redSlider:ScrollBar;
		private var greenSlider:ScrollBar;
		private var blueSlider:ScrollBar;
		
		public function ControlPanel(particleSystem:ParticleSystem){
			this.particleSystem = particleSystem;
			
			attractionRadiusMinSlider = new ScrollBar("Attraction Radius Min", 10, 10, 25, 500, 250, attractionRadiusMinCallback);
			addChild(attractionRadiusMinSlider);
			attractionRadiusMaxSlider = new ScrollBar("Attraction Radius Max", 10, 40, 25, 500, 250, attractionRadiusMaxCallback);
			addChild(attractionRadiusMaxSlider);
			repulsionRadiusSlider = new ScrollBar("Repulsion Radius", 10, 70, 0, 100, 50, repulsionRadiusCallback);
			addChild(repulsionRadiusSlider);
			massSlider = new ScrollBar("Mass", 10, 100, 0, 5, 1, massCallback);
			addChild(massSlider);
			torqueSlider = new ScrollBar("Torque", 300, 10, -2, 2, -0.1, torqueCallback);
			addChild(torqueSlider);
			redSlider = new ScrollBar("Red", 300, 40, 0, 255, 1, redCallback);
			addChild(redSlider);
			greenSlider = new ScrollBar("Green", 300, 70, 0, 255, 1, greenCallback);
			addChild(greenSlider);
			blueSlider = new ScrollBar("Blue", 300, 100, 0, 255, 1, blueCallback);
			addChild(blueSlider);			
			
			this.addEventListener(MouseEvent.MOUSE_UP, mouseReleased);
			alpha = 0.5;
		}
		
		public function mouseReleased(e:MouseEvent):void{
			trace("mouse released");
			// make sure all sliders are set to mouseDown in case it was released outside
			attractionRadiusMaxSlider.mouseDown = false;
			attractionRadiusMinSlider.mouseDown = false;
			repulsionRadiusSlider.mouseDown = false;
			massSlider.mouseDown = false;
			torqueSlider.mouseDown = false;
			redSlider.mouseDown = false;
			greenSlider.mouseDown = false;
			blueSlider.mouseDown = false;
		}
		
		
		
		
		public function updateValues(attractionRadiusMin:Number, attractionRadiusMax:Number, repulsionRadius:Number,
									 mass:Number, torque:Number, red:Number, green:Number, blue:Number):void{
			attractionRadiusMinSlider.setValue(attractionRadiusMin);
			attractionRadiusMaxSlider.setValue(attractionRadiusMax);
			repulsionRadiusSlider.setValue(repulsionRadius);
			massSlider.setValue(mass);
			torqueSlider.setValue(torque);
			redSlider.setValue(red);
			greenSlider.setValue(green);
			blueSlider.setValue(blue);
		}
		
		
		
		public function attractionRadiusMaxCallback(val:Number):void{
			particleSystem.setRadiusOfAttractionMax(val);
		}
		
		public function attractionRadiusMinCallback(val:Number):void{
			particleSystem.setRadiusOfAttractionMin(val);
		}
		
		public function repulsionRadiusCallback(val:Number):void{
			particleSystem.setRadiusOfRepulsion(val);
		}
		
		public function massCallback(val:Number):void{
			particleSystem.setMass(val);
		}
		
		public function torqueCallback(val:Number):void{
			particleSystem.setTorque(val);
		}
		
		public function redCallback(val:Number):void{
			particleSystem.setRed(val);
		}
		
		public function greenCallback(val:Number):void{
			particleSystem.setGreen(val);
		}
		
		public function blueCallback(val:Number):void{
			particleSystem.setBlue(val);
		}
		
	}
	
}