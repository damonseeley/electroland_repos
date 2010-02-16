package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	import flash.events.Event;
	import flash.events.KeyboardEvent;
	import flash.text.TextField;
	import flash.text.TextFormat;
	
	public class ControlPanel extends MovieClip{
		
		private var particleSystem:ParticleSystem;
		private var macroManager:MacroManager;
		private var basicControls:MovieClip;
		private var advancedControls:MovieClip;
		
		// text labels for individual control sections
		private var motionLabel:TextField;
		private var shapeLabel:TextField;
		private var sizeLabel:TextField;
		private var colorLabel:TextField;
		private var presetsLabel:TextField;
		private var radiusLabel:TextField;
		private var rotationLabel:TextField;
		private var speedLabel:TextField;
		private var particlesLabel:TextField;
		private var sparksLabel:TextField;
		
		// scrollbars
		private var attractionRadiusMaxSlider:ScrollBar;
		private var attractionRadiusMinSlider:ScrollBar;
		private var repulsionRadiusSlider:ScrollBar;
		private var massSlider:ScrollBar;
		private var atomicSpeedSlider:ScrollBar;
		private var springSpeedSlider:ScrollBar;
		private var torqueSlider:ScrollBar;
		private var redSlider:ScrollBar;
		private var greenSlider:ScrollBar;
		private var blueSlider:ScrollBar;
		private var hueSlider:ScrollBar;
		private var particleMinSizeSlider:ScrollBar;
		private var particleMaxSizeSlider:ScrollBar;
		private var particleMinSpinSlider:ScrollBar;
		private var particleMaxSpinSlider:ScrollBar;
		private var particleCountSlider:ScrollBar;
		private var sparksSpeedSlider:ScrollBar;
		private var sparksLifeMinSlider:ScrollBar;
		private var sparksLifeMaxSlider:ScrollBar;
		private var sparksEmitterDelay:ScrollBar;
		
		// radio button groups
		private var visualMode:RadioButtonGroup;
		private var gravityMode:RadioButtonGroup;
		private var presetMode:RadioButtonGroup;
		
		// text stuff
		private var particleCount:TextField;
		private var textFormat:TextFormat;
		
		public function ControlPanel(particleSystem:ParticleSystem){
			this.particleSystem = particleSystem;
			basicControls = new MovieClip();
			addChild(basicControls);
			basicControls.graphics.lineStyle(1, 0x666666, 0.5);
			basicControls.graphics.moveTo(90,0);		// vertical line
			basicControls.graphics.lineTo(90,125);
			
			basicControls.graphics.moveTo(290,95);
			basicControls.graphics.lineTo(290,125);
			basicControls.graphics.moveTo(590,95);
			basicControls.graphics.lineTo(590,125);
			
			basicControls.graphics.moveTo(0,35);		// horizontal lines
			basicControls.graphics.lineTo(1024,35);
			basicControls.graphics.moveTo(0,65);
			basicControls.graphics.lineTo(1024,65);
			basicControls.graphics.moveTo(0,95);
			basicControls.graphics.lineTo(1024,95);
			basicControls.graphics.moveTo(0,125);
			basicControls.graphics.lineTo(1024,125);
			//basicControls.graphics.moveTo(0,155);
			//basicControls.graphics.lineTo(1024,155);
			//basicControls.graphics.moveTo(0,185);
			//basicControls.graphics.lineTo(1024,185);
			
			var titles = new BasicControlsTitles();
			basicControls.addChild(titles);
			titles.y = 10;
			titles.x = 5;
			
			textFormat = new TextFormat("Verdana", 12, 0x999999);
			textFormat.bold = true;
			
			
			// PRESETS SECTION
			/*
			presetsLabel = new TextField();
			presetsLabel.text = "PRESETS:";
			presetsLabel.x = 5;
			presetsLabel.y = 10;
			presetsLabel.autoSize = "left";
			presetsLabel.selectable = false;
			presetsLabel.setTextFormat(textFormat);
			basicControls.addChild(presetsLabel);	
			*/
			presetMode = new RadioButtonGroup(100, 10, presetModeCallback);
			presetMode.addButton(1, "1", 0, 0, true);
			presetMode.addButton(2, "2", 50, 0, false);
			presetMode.addButton(3, "3", 100, 0, false);
			presetMode.addButton(4, "4", 150, 0, false);
			presetMode.addButton(5, "5", 200, 0, false);
			basicControls.addChild(presetMode);
			
			// MOTION SECTION
			/*
			motionLabel = new TextField();
			motionLabel.text = "MOTION:";
			motionLabel.x = 5;
			motionLabel.y = 40;
			motionLabel.autoSize = "left";
			motionLabel.selectable = false;
			motionLabel.setTextFormat(textFormat);
			basicControls.addChild(motionLabel);
			*/
			gravityMode = new RadioButtonGroup(100, 40, gravityModeCallback);
			gravityMode.addButton(0, "Gravity", 0, 0, true);
			gravityMode.addButton(1, "Square", 100, 0, false);
			//gravityMode.addButton(2, "Star", 0, 10, false);
			gravityMode.addButton(3, "Spring", 200, 0, false);
			gravityMode.addButton(4, "Atomic", 300, 0, false);
			gravityMode.addButton(5, "Sparks", 400, 0, false);
			basicControls.addChild(gravityMode);
			
			// SHAPE SECTION
			/*
			shapeLabel = new TextField();
			shapeLabel.text = "SHAPE:";
			shapeLabel.x = 5;
			shapeLabel.y = 70;
			shapeLabel.autoSize = "left";
			shapeLabel.selectable = false;
			shapeLabel.setTextFormat(textFormat);
			basicControls.addChild(shapeLabel);
			*/
			visualMode = new RadioButtonGroup(100, 70, visualModeCallback);
			visualMode.addButton(0, "Circle", 0, 0, true);
			visualMode.addButton(1, "Green Dot", 100, 0, false);
			visualMode.addButton(2, "Blue Dot", 200, 0, false);
			visualMode.addButton(3, "Hexagon", 300, 0, false);
			visualMode.addButton(4, "Cross", 400, 0, false);
			visualMode.addButton(5, "Line", 500, 0, false);
			visualMode.addButton(6, "Rounded Rectangle", 600, 0, true);
			basicControls.addChild(visualMode);
			
			// SIZE SECTION
			/*
			sizeLabel = new TextField();
			sizeLabel.text = "SIZE:";
			sizeLabel.x = 5;
			sizeLabel.y = 100;
			sizeLabel.autoSize = "left";
			sizeLabel.selectable = false;
			sizeLabel.setTextFormat(textFormat);
			basicControls.addChild(sizeLabel);	
			*/
			particleMinSizeSlider = new ScrollBar("Min", 100, 100, 1, 50, 3, particleMinSizeCallback, 0x666666);
			//basicControls.addChild(particleMinSizeSlider);
			particleMaxSizeSlider = new ScrollBar("Particle Size", 100, 100, 1, 50, 6, particleMaxSizeCallback, 0x666666);
			basicControls.addChild(particleMaxSizeSlider);	
			
			// COLOR SECTION
			/*
			colorLabel = new TextField();
			colorLabel.text = "COLOR:";
			colorLabel.x = 300;
			colorLabel.y = 100;
			colorLabel.autoSize = "left";
			colorLabel.selectable = false;
			colorLabel.setTextFormat(textFormat);
			basicControls.addChild(colorLabel);	
			*/
			redSlider = new ScrollBar("Red", 100, 130, 0, 255, 1, redCallback, 0xff0000);
			//basicControls.addChild(redSlider);
			greenSlider = new ScrollBar("Green", 250, 130, 0, 255, 1, greenCallback, 0x00ff00);
			//basicControls.addChild(greenSlider);
			blueSlider = new ScrollBar("Blue", 400, 130, 0, 255, 1, blueCallback, 0x0000ff);
			//basicControls.addChild(blueSlider);
			hueSlider = new ScrollBar("Hue", 400, 100, 0, 360, 1, hueCallback, 0x666666);
			basicControls.addChild(hueSlider);
			
			
			// PARTICLE SECTION
			/*
			particlesLabel = new TextField();
			particlesLabel.text = "DENSITY:";
			particlesLabel.x = 600;
			particlesLabel.y = 100;
			particlesLabel.autoSize = "left";
			particlesLabel.selectable = false;
			particlesLabel.setTextFormat(textFormat);
			basicControls.addChild(particlesLabel);
			*/
			particleCountSlider = new ScrollBar("Particle Count", 700, 100, 0, 100, 20, particleCountCallback, 0x666666);
			basicControls.addChild(particleCountSlider);
			
			//textFormat = new TextFormat("Arial", 10, 0x333333);
			particleCount = new TextField();
			particleCount.text = "Total: "+ String(particleSystem.particles.size());
			particleCount.x = 300;
			particleCount.y = 100;
			particleCount.autoSize = "left";
			particleCount.selectable = false;
			particleCount.setTextFormat(new TextFormat("Arial", 12, 0x333333));
			//advancedControls.addChild(particleCount);
			
			
			
			advancedControls = new MovieClip();
			addChild(advancedControls);
			advancedControls.y = 120;
			advancedControls.visible = false;
			advancedControls.graphics.lineStyle(1, 0x666666, 0.5);
			advancedControls.graphics.moveTo(90,5);		// vertical line
			advancedControls.graphics.lineTo(90,135);
			advancedControls.graphics.moveTo(0,35);		// horizontal lines
			advancedControls.graphics.lineTo(1024,35);
			advancedControls.graphics.moveTo(0,65);
			advancedControls.graphics.lineTo(1024,65);
			advancedControls.graphics.moveTo(0,95);
			advancedControls.graphics.lineTo(1024,95);
			advancedControls.graphics.moveTo(0,125);
			advancedControls.graphics.lineTo(1024,125);
			
			var advancedTitles = new AdvancedControlsTitles();
			advancedControls.addChild(advancedTitles);
			advancedTitles.x = 5;
			advancedTitles.y = 10;
			
			// RADIUS SECTION
			/*
			radiusLabel = new TextField();
			radiusLabel.text = "RADIUS:";
			radiusLabel.x = 5;
			radiusLabel.y = 10;
			radiusLabel.autoSize = "left";
			radiusLabel.selectable = false;
			radiusLabel.setTextFormat(textFormat);
			advancedControls.addChild(radiusLabel);
			*/
			attractionRadiusMinSlider = new ScrollBar("Attraction Min", 100, 10, 25, 500, 250, attractionRadiusMinCallback, 0x666666);
			advancedControls.addChild(attractionRadiusMinSlider);
			attractionRadiusMaxSlider = new ScrollBar("Attraction Max", 300, 10, 25, 500, 250, attractionRadiusMaxCallback, 0x666666);
			advancedControls.addChild(attractionRadiusMaxSlider);
			repulsionRadiusSlider = new ScrollBar("Repulsion Max", 500, 10, 0, 100, 50, repulsionRadiusCallback, 0x666666);
			advancedControls.addChild(repulsionRadiusSlider);
			
			
			// ROTATION SECTION
			/*
			rotationLabel = new TextField();
			rotationLabel.text = "ROTATION:";
			rotationLabel.x = 5;
			rotationLabel.y = 40;
			rotationLabel.autoSize = "left";
			rotationLabel.selectable = false;
			rotationLabel.setTextFormat(textFormat);
			advancedControls.addChild(rotationLabel);
			*/
			
			torqueSlider = new ScrollBar("Person Torque", 100, 40, -2, 2, -0.1, torqueCallback, 0x666666);
			advancedControls.addChild(torqueSlider);
			particleMinSpinSlider = new ScrollBar("Particle Min Spin", 300, 40, -20, 0, -2, particleMinSpinCallback, 0x666666);
			advancedControls.addChild(particleMinSpinSlider);
			particleMaxSpinSlider = new ScrollBar("Particle Max Spin", 500, 40, 0, 20, 2, particleMaxSpinCallback, 0x666666);
			advancedControls.addChild(particleMaxSpinSlider);	
			
			
			// SPEED SECTION
			/*
			speedLabel = new TextField();
			speedLabel.text = "SPEED:";
			speedLabel.x = 5;
			speedLabel.y = 70;
			speedLabel.autoSize = "left";
			speedLabel.selectable = false;
			speedLabel.setTextFormat(textFormat);
			advancedControls.addChild(speedLabel);
			*/
			massSlider = new ScrollBar("Mass", 100, 70, 0, 5, 1, massCallback, 0x666666);
			advancedControls.addChild(massSlider);
			springSpeedSlider = new ScrollBar("Spring Speed", 300, 70, 0.1, 10, 2, springSpeedCallback, 0x666666);
			advancedControls.addChild(springSpeedSlider);
			atomicSpeedSlider = new ScrollBar("Atomic Speed", 500, 70, 0.1, 10, 2, atomicSpeedCallback, 0x666666);
			advancedControls.addChild(atomicSpeedSlider);
			
		
			
			
			// SPARKS SECTION
			/*
			sparksLabel = new TextField();
			sparksLabel.text = "SPARKS:";
			sparksLabel.x = 5;
			sparksLabel.y = 100;
			sparksLabel.autoSize = "left";
			sparksLabel.selectable = false;
			sparksLabel.setTextFormat(textFormat);
			advancedControls.addChild(sparksLabel);
			*/
			sparksSpeedSlider = new ScrollBar("Speed", 100, 100, 0.1, 20, 2, sparksSpeedCallback, 0x666666);
			advancedControls.addChild(sparksSpeedSlider);
			sparksLifeMinSlider = new ScrollBar("Life Min", 300, 100, 0.1, 5, 2, sparksLifeMinCallback, 0x666666);
			advancedControls.addChild(sparksLifeMinSlider);
			sparksLifeMaxSlider = new ScrollBar("Life Max", 500, 100, 0.1, 5, 2, sparksLifeMaxCallback, 0x666666);
			advancedControls.addChild(sparksLifeMaxSlider);
			sparksEmitterDelay = new ScrollBar("Emitter Delay", 700, 100, 1, 1000, 50, sparksEmitterDelayCallback, 0x666666);
			advancedControls.addChild(sparksEmitterDelay);
			
			this.addEventListener(MouseEvent.MOUSE_UP, mouseReleased);
			this.addEventListener(Event.ENTER_FRAME, countParticles);
			alpha = 0.5;
		}
		
		public function setup(macroManager:MacroManager):void{
			this.macroManager = macroManager;
			stage.addEventListener(KeyboardEvent.KEY_DOWN,keyDownListener);
		}
		
		public function keyDownListener(e:KeyboardEvent):void{
			if(e.keyCode == 75){ // k
				if(advancedControls.visible){
					advancedControls.visible = false;
				} else {
					advancedControls.visible = true;
				}
			}
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
		
		public function countParticles(e:Event){
			particleCount.text = "Total: "+ String(particleSystem.particles.size());
			particleCount.setTextFormat(new TextFormat("Arial", 12, 0x333333));
		}

		
		
		
		
		public function updateValues(attractionRadiusMin:Number, attractionRadiusMax:Number, repulsionRadius:Number,
									 mass:Number, torque:Number, red:Number, green:Number, blue:Number, minRadius:Number,
									 maxRadius:Number, minSpin:Number, maxSpin:Number, visualModeNum:Number, gravityModeNum:Number,
									 atomicSpeed:Number, springSpeed:Number, particleCount:Number, sparkSpeed:Number, sparkLifeMin:Number,
									 sparkLifeMax:Number, emitterDelay:Number):void{
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
			atomicSpeedSlider.setValue(atomicSpeed);
			springSpeedSlider.setValue(springSpeed);
			particleCountSlider.setValue(particleCount);
			sparksSpeedSlider.setValue(sparkSpeed);
			sparksLifeMinSlider.setValue(sparkLifeMin);
			sparksLifeMaxSlider.setValue(sparkLifeMax);
			sparksEmitterDelay.setValue(emitterDelay);
			
			var hsv:Array = RGBtoHSV(red, green, blue);
			hueSlider.setValue(hsv[0]);
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
							gravityMode.getValue(),
							atomicSpeedSlider.getValue(),
							springSpeedSlider.getValue(),
							particleCountSlider.getValue(),
							sparksSpeedSlider.getValue(),
							sparksLifeMinSlider.getValue(),
							sparksLifeMaxSlider.getValue(),
							sparksEmitterDelay.getValue());
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
			atomicSpeedSlider.setValue(macro.atomicSpeed);
			particleSystem.setAtomicSpeed(macro.atomicSpeed);
			springSpeedSlider.setValue(macro.springSpeed);
			particleSystem.setSpringSpeed(macro.springSpeed);
			particleCountSlider.setValue(macro.particleCount);
			particleSystem.setParticleCount(macro.particleCount);
			sparksSpeedSlider.setValue(macro.sparkSpeed);
			particleSystem.setSparkSpeed(macro.sparkSpeed);
			sparksLifeMinSlider.setValue(macro.sparkLifeMin);
			particleSystem.setSparkLifeMin(macro.sparkLifeMin);
			sparksLifeMaxSlider.setValue(macro.sparkLifeMax);
			particleSystem.setSparkLifeMax(macro.sparkLifeMax);
			sparksEmitterDelay.setValue(macro.emitterDelay);
			particleSystem.setSparkEmitterDelay(macro.emitterDelay);
			
			var hsv:Array = RGBtoHSV(macro.red, macro.green, macro.blue);
			hueSlider.setValue(hsv[0]);
			
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
		
		public function hueCallback(val:Number):void{
			particleSystem.setHue(val);
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
		
		public function presetModeCallback(val:Number):void{
			macroManager.applyMacro(val);			
		}
		
		public function springSpeedCallback(val:Number):void{
			particleSystem.setSpringSpeed(val);
		}
		
		public function atomicSpeedCallback(val:Number):void{
			particleSystem.setAtomicSpeed(val);
		}
		
		public function particleCountCallback(val:Number):void{
			particleSystem.setParticleCount(val);
		}
		
		public function sparksSpeedCallback(val:Number):void{
			particleSystem.setSparkSpeed(val);
		}
		
		public function sparksLifeMinCallback(val:Number):void{
			particleSystem.setSparkLifeMin(val);
		}
		
		public function sparksLifeMaxCallback(val:Number):void{
			particleSystem.setSparkLifeMax(val);
		}
		
		public function sparksEmitterDelayCallback(val:Number):void{
			particleSystem.setSparkEmitterDelay(val);
		}
		
		
		// color utility
		
		private function RGBtoHSV(r:uint, g:uint, b:uint):Array{
			var max:uint = Math.max(r, g, b);
			var min:uint = Math.min(r, g, b);
			var hue:Number = 0;
			var saturation:Number = 0;
			var value:Number = 0;
	
			var hsv:Array = [];
	
			 //get Hue
			if(max == min){
				hue = 0;
			}else if(max == r){
				hue = (60 * (g-b) / (max-min) + 360) % 360;
			}else if(max == g){
				hue = (60 * (b-r) / (max-min) + 120);
			}else if(max == b){
				hue = (60 * (r-g) / (max-min) + 240);
			}
	
			//get Value
			value = max;
			//get Saturation
			if(max == 0){
				saturation = 0;
			}else{
				saturation = (max - min) / max;
			}
	
			hsv = [Math.round(hue), Math.round(saturation * 100), Math.round(value / 255 * 100)];
			return hsv;
		}
		
	}
	
}