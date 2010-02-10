package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	
	public class ControlPanel extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		
		private var attractionRadiusMaxSlider:ScrollBar;
		private var attractionRadiusMinSlider:ScrollBar;
		private var repulsionRadiusSlider:ScrollBar;
		private var massSlider:ScrollBar;
		private var torqueSlider:ScrollBar;
		private var redSlider:ScrollBar;
		private var greenSlider:ScrollBar;
		private var blueSlider:ScrollBar;
		private var particleMinSizeSlider:ScrollBar;
		private var particleMaxSizeSlider:ScrollBar;
		private var particleMinSpinSlider:ScrollBar;
		private var particleMaxSpinSlider:ScrollBar;
		private var visualMode:RadioButtonGroup;
		private var gravityMode:RadioButtonGroup;
		
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
			particleMinSizeSlider = new ScrollBar("Particle Min Radius", 500, 10, 1, 50, 3, particleMinSizeCallback);
			addChild(particleMinSizeSlider);
			particleMaxSizeSlider = new ScrollBar("Particle Max Radius", 500, 40, 1, 50, 6, particleMaxSizeCallback);
			addChild(particleMaxSizeSlider);	
			particleMinSpinSlider = new ScrollBar("Particle Min Spin", 500, 70, -20, 0, -2, particleMinSpinCallback);
			addChild(particleMinSpinSlider);
			particleMaxSpinSlider = new ScrollBar("Particle Max Spin", 500, 100, 0, 20, 2, particleMaxSpinCallback);
			addChild(particleMaxSpinSlider);	
			
			visualMode = new RadioButtonGroup(850, 0, visualModeCallback);
			visualMode.addButton(0, "Circle", 0, 10, true);
			visualMode.addButton(1, "Green Dot", 0, 40, false);
			visualMode.addButton(2, "Blue Dot", 0, 70, false);
			visualMode.addButton(3, "Hexagon", 0, 100, false);
			visualMode.addButton(4, "Cross", 0, 130, false);
			visualMode.addButton(5, "Line", 0, 160, false);
			visualMode.addButton(6, "Rounded Rectangle", 0, 190, true);
			//visualMode.addButton(0, "Circle", 0, 10, true);
			//visualMode.addButton(1, "Soft", 0, 40, false);
			//visualMode.addButton(2, "Jagged", 0, 70, false);
			//visualMode.addButton(3, "Line", 0, 100, false);
			addChild(visualMode);
			
			gravityMode = new RadioButtonGroup(750, 0, gravityModeCallback);
			gravityMode.addButton(0, "Gravity", 0, 10, true);
			//gravityMode.addButton(1, "Square", 0, 40, false);
			//gravityMode.addButton(2, "Star", 0, 70, false);
			gravityMode.addButton(3, "Spring", 0, 40, false);
			gravityMode.addButton(4, "Atomic", 0, 70, false);
			addChild(gravityMode);
			
			this.addEventListener(MouseEvent.MOUSE_UP, mouseReleased);
			alpha = 0.5;
		}
		
		public function mouseReleased(e:MouseEvent):void{
			//trace("mouse released");
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
									 mass:Number, torque:Number, red:Number, green:Number, blue:Number, minRadius:Number,
									 maxRadius:Number, minSpin:Number, maxSpin:Number, visualModeNum:Number, gravityModeNum:Number):void{
			attractionRadiusMinSlider.setValue(attractionRadiusMin);
			attractionRadiusMaxSlider.setValue(attractionRadiusMax);
			repulsionRadiusSlider.setValue(repulsionRadius);
			massSlider.setValue(mass);
			torqueSlider.setValue(torque);
			redSlider.setValue(red);
			greenSlider.setValue(green);
			blueSlider.setValue(blue);
			particleMinSizeSlider.setValue(minRadius);
			particleMaxSizeSlider.setValue(maxRadius);
			particleMinSpinSlider.setValue(minSpin);
			particleMaxSpinSlider.setValue(maxSpin);
			visualMode.activate(visualModeNum);
			gravityMode.activate(gravityModeNum);
		}
		
		public function getMacro(id:Number):Macro{
			var macro:Macro = new Macro(id);
			macro.setValues(attractionRadiusMinSlider.getValue(),
							attractionRadiusMaxSlider.getValue(),
							repulsionRadiusSlider.getValue(),
							massSlider.getValue(),
							torqueSlider.getValue(),
							redSlider.getValue(),
							greenSlider.getValue(),
							blueSlider.getValue(),
							particleMinSizeSlider.getValue(),
							particleMaxSizeSlider.getValue(),
							particleMinSpinSlider.getValue(),
							particleMaxSpinSlider.getValue(),
							visualMode.getValue(),
							gravityMode.getValue());
			return macro;
		}
		
		public function loadMacro(macro:Macro):void{
			attractionRadiusMinSlider.setValue(macro.attractionRadiusMin);
			particleSystem.setRadiusOfAttractionMin(macro.attractionRadiusMin);
			attractionRadiusMaxSlider.setValue(macro.attractionRadiusMax);
			particleSystem.setRadiusOfAttractionMax(macro.attractionRadiusMax);
			repulsionRadiusSlider.setValue(macro.repulsionRadius);
			particleSystem.setRadiusOfRepulsion(macro.repulsionRadius);
			massSlider.setValue(macro.mass);
			particleSystem.setMass(macro.mass);
			torqueSlider.setValue(macro.torque);
			particleSystem.setTorque(macro.torque);
			redSlider.setValue(macro.red);
			particleSystem.setRed(macro.red);
			greenSlider.setValue(macro.green);
			particleSystem.setGreen(macro.green);
			blueSlider.setValue(macro.blue);
			particleSystem.setBlue(macro.blue);
			particleMinSizeSlider.setValue(macro.minRadius);
			particleSystem.setParticleMinSize(macro.minRadius);
			particleMaxSizeSlider.setValue(macro.maxRadius);
			particleSystem.setParticleMaxSize(macro.maxRadius);
			particleMinSpinSlider.setValue(macro.minSpin);
			particleSystem.setParticleMinSpin(macro.minSpin);
			particleMaxSpinSlider.setValue(macro.maxSpin);
			visualMode.activate(macro.visualMode);
			particleSystem.setVisualMode(macro.visualMode);
			gravityMode.activate(macro.gravityMode);
			particleSystem.setGravityMode(macro.gravityMode);
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
		
		public function particleMinSizeCallback(val:Number):void{
			particleSystem.setParticleMinSize(val);
		}
		
		public function particleMaxSizeCallback(val:Number):void{
			particleSystem.setParticleMaxSize(val);
		}
		
		public function particleMinSpinCallback(val:Number):void{
			particleSystem.setParticleMinSpin(val);
		}
		
		public function particleMaxSpinCallback(val:Number):void{
			particleSystem.setParticleMaxSpin(val);
		}
		
		public function visualModeCallback(val:Number):void{
			particleSystem.setVisualMode(val);
		}
		
		public function gravityModeCallback(val:Number):void{
			particleSystem.setGravityMode(val);
		}
		
	}
	
}