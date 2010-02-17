package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.Event;
	import flash.events.KeyboardEvent;
	import flash.events.MouseEvent;
	import flash.display.Loader;
	import flash.display.LoaderInfo;
	import flash.net.URLRequest;
	import com.ericfeminella.collections.HashMap;
	
	public class ParticleSystem extends MovieClip{
		
		//public var people:Array;				// container of person objects
		public var people:HashMap;
		public var particles:HashMap;			// container of particle objects
		private var personID:Number;			// unique ID for each person
		private var particleID:Number;			// unique ID for each particle
		private var bgLayer:MovieClip;
		private var particleLayer:MovieClip;
		private var personLayer:MovieClip;
		private var colors:Array;				// preset colors
		public var selectedPerson:Number;		// id of selected person
		private var controlPanel:ControlPanel;	// reference just for updating values
		public var softParticle:Loader;			// images for particles
		public var weirdParticle:Loader;
		public var lineParticle:Loader;
		
		/*
		public var redDot:Loader;				// new images
		public var greenDot:Loader;
		public var blueDot:Loader;
		public var cross:Loader;
		public var hexagon:Loader;
		public var roundRect:Loader;
		*/
		public var addPerson:Boolean = false;	// set to true when CTRL is held
		
		/*
		PARTICLESYSTEM.as
		by Aaron Siegel, 2-1-2010
		
		Controls updating the state of the system as a whole, including adding or
		removing Person objects or Particle objects.
		*/
		
		public function ParticleSystem(){
			bgLayer = new MovieClip();
			particleLayer = new MovieClip();
			personLayer = new MovieClip();
			addChild(bgLayer);
			addChild(particleLayer);
			addChild(personLayer);
			
			colors = new Array();
			var color1:Array = new Array();	// cyan
			color1.push(0);
			color1.push(150);
			color1.push(255);
			color1.push(0.8);
			var color2:Array = new Array();	// orange
			color2.push(255);
			color2.push(100);
			color2.push(0);
			color2.push(0.8);
			var color3:Array = new Array();	// green
			color3.push(0);
			color3.push(200);
			color3.push(0);
			color3.push(0.8);
			
			colors.push(color1);	// preset colors
			colors.push(color2);
			colors.push(color3);
		}
		
		public function setup(particleCount:Number, personCount:Number):void{
			personID = 0;
			particleID = 0;
			
			bgLayer.graphics.beginFill(0x000000);
			bgLayer.graphics.drawRect(0,0,stage.stageWidth,stage.stageHeight);
			bgLayer.graphics.endFill();
			bgLayer.alpha = 0;

			//softParticle = new Loader();
			//softParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			//softParticle.load(new URLRequest("images/softParticle.png"));
			//weirdParticle = new Loader();
			//weirdParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			//weirdParticle.load(new URLRequest("images/weirdParticle.png"));
			/*
			lineParticle = new Loader();
			lineParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			lineParticle.load(new URLRequest("images/line.png"));
			redDot = new Loader();
			redDot.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			redDot.load(new URLRequest("images/reddot.png"));
			greenDot = new Loader();
			greenDot.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			greenDot.load(new URLRequest("images/greendot.png"));
			blueDot = new Loader();
			blueDot.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			blueDot.load(new URLRequest("images/bluedot.png"));
			cross = new Loader();
			cross.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			cross.load(new URLRequest("images/cross.png"));
			hexagon = new Loader();
			hexagon.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			hexagon.load(new URLRequest("images/hexagon.png"));
			roundRect = new Loader();
			roundRect.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			roundRect.load(new URLRequest("images/roundrect.png"));
			*/
			
			
			//people = new Array();
			people = new HashMap();
			particles = new HashMap();
			for(var i:Number = 0; i<personCount; i++){
				var xPos:Number = Math.random()*stage.stageWidth;
				var yPos:Number = Math.random()*stage.stageHeight;
				if(personCount == 1){
					xPos = stage.stageWidth / 2;	// center person if only using one
					yPos = stage.stageHeight / 2;
				}
				var radius:Number = 25;
				var mass:Number = 1;
				var torque:Number = (-0.1 * Math.random()) - 0.05;	// counter clockwise
				var person:Person = new Person(personID, xPos, yPos, radius, mass, torque);
				person.setParticleColor(colors[i]);
				personLayer.addChild(person);
				person.addCallback(this);
				//people.push(person);
				people.put(personID, person);
				if(personCount == 1){
					person.silentSelect();
					selectedPerson = personID;
				}
				for(var p:Number = 0; p<particleCount; p++){
					createNewParticle(personID, xPos, yPos, -2 + (Math.random() * 4), 0);
				}
				personID++;
			}
			
			this.addEventListener(Event.ENTER_FRAME, update);
			stage.addEventListener(KeyboardEvent.KEY_DOWN,keyDownListener);
			stage.addEventListener(KeyboardEvent.KEY_UP,keyUpListener);
			bgLayer.addEventListener(MouseEvent.MOUSE_DOWN, mouseDownListener);
		}
		
		public function imageLoaded(e:Event):void{
			//trace(e.target.content.parent + " loaded");
			//info.target.content.width = radius*2;
			//info.target.content.height = radius*2;
		}
		
		public function update(e:Event):void{
			
			var values:Array = particles.getValues();
			for(var i:Number = 0; i<values.length; i++){
				values[i].move();
			}
			for(i = 0; i<values.length; i++){
				values[i].applyRotation(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applySquareRotation(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applyGravity(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applySquareGravity(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applyStarGravity(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applySpringGravity(people.getValues());
			}
			for(i = 0; i<values.length; i++){
				values[i].applyAtomicGravity(people.getValues());
			}
		}
		
		public function createNumParticles(num:Number, emitterID:Number, xPos:Number, yPos:Number):void{
			for(var p:Number = 0; p<num; p++){
				createNewParticle(emitterID, xPos, yPos, -2 + (Math.random() * 4), people.getValue(emitterID).visualMode);
			}
		}
		
		public function createNewParticlesOutside(emitterID:Number, xPos:Number, yPos:Number):void{
			for(var p:Number = 0; p<people.getValue(emitterID).particleCount; p++){
				xPos += (Math.random()*100) - 50;
				yPos += (Math.random()*100) - 50;
				createNewParticle(emitterID, xPos, yPos, -2 + (Math.random() * 4), people.getValue(emitterID).visualMode);
			}
		}
		
		public function createNewParticles(emitterID:Number, xPos:Number, yPos:Number):void{
			for(var p:Number = 0; p<people.getValue(emitterID).particleCount; p++){
				createNewParticle(emitterID, xPos, yPos, -2 + (Math.random() * 4), people.getValue(emitterID).visualMode);
			}
		}
		
		public function createNewParticle(emitterID:Number, xPos:Number, yPos:Number, spin:Number, visualMode:Number):void{
			// emit particles from this point with an initial random vector
			if(people.containsKey(emitterID)){
				var mass:Number = Math.random() + 0.1;	// 0.1 - 1
				var scale:Number = Math.random();
				var minRadius:Number = people.getValue(emitterID).particleMinRadius;//people[emitterID].particleMinRadius;
				var maxRadius:Number = people.getValue(emitterID).particleMaxRadius;//people[emitterID].particleMaxRadius;
				var particle:Particle = new Particle(particleID, emitterID, xPos, yPos, scale, minRadius, maxRadius, mass, spin, visualMode, this);
				if(visualMode < 1){
					//particle.setColor(people[emitterID].getParticleColor());
					particle.setColor(people.getValue(emitterID).getParticleColor());
				}
				particleLayer.addChild(particle);
				particles.put(particleID, particle);
				particleID++;
			}
		}
		
		public function removeParticle(e:ParticleEvent):void{
			if(particles.containsKey(e.id)){
				if(particleLayer.contains(particles.getValue(e.id))){
					particleLayer.removeChild(particles.getValue(e.id));
				}
				particles.remove(e.id);	// remove particle
			}
			//trace("particle "+ e.id +" removed, "+ particles.size() + " left");
		}
		
		public function removePerson(id:Number):void{
			personLayer.removeChild(people.getValue(id));
			people.remove(id);
		}
		
		public function personSelected(id:Number):void{
			selectedPerson = id;
			//for(var i:Number = 0; i<people.length; i++){
				//if(people[i].id != id){
					//people[i].deselect();
				//}
			//}
			var values:Array = people.getValues();
			for(var i:Number = 0; i<values.length; i++){
				if(values[i].id != id){
					values[i].deselect();
				}
			}
			// update control panel
			var person = people.getValue(id);
			controlPanel.updateValues(person.radiusOfAttractionMin, person.radiusOfAttractionMax, person.radiusOfRepulsion,
									  person.mass, person.torque, person.particleColorRed, person.particleColorGreen,
									  person.particleColorBlue, person.particleMinRadius, person.particleMaxRadius,
									  person.particleSpinMin, person.particleSpinMax, person.visualMode, person.gravityMode,
									  person.atomicSpeed, person.springSpeed, person.particleCount, person.sparkSpeed, person.sparkLifeMin,
									  person.sparkLifeMax, person.emitterSpeed);
		}
		
		
		
		// FUNCTIONS FOR MODIFYING PERSON AND PROPERTY VALUES ON THE FLY
		
		public function addControlPanel(controlPanel:ControlPanel):void{
			this.controlPanel = controlPanel;
		}
		
		public function setRadiusOfAttractionMax(radiusOfAttractionMax:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setRadiusOfAttractionMax(radiusOfAttractionMax);
			}
		}
		
		public function setRadiusOfAttractionMin(radiusOfAttractionMin:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setRadiusOfAttractionMin(radiusOfAttractionMin);
			}
		}
		
		public function setRadiusOfRepulsion(radiusOfRepulsion:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setRadiusOfRepulsion(radiusOfRepulsion);
			}
		} 
		
		public function setTorque(torque:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setTorque(torque);
			}
		}
		
		public function setMass(mass:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setMass(mass);
			}
		}
		
		public function setRed(red:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setRed(red);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setRed(red);
					}
				}
			}
		}
		
		public function setGreen(green:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setGreen(green);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setGreen(green);
					}
				}
			}
		}
		
		public function setBlue(blue:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setBlue(blue);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setBlue(blue);
					}
				}
			}
		}
		
		public function setHue(hue:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setHue(hue);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setHue(hue);
					}
				}
			}
		}
		
		public function setParticleMinSize(size:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setParticleMinSize(size);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setMinRadius(size);
					}
				}
			}
		}
		
		public function setParticleMaxSize(size:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setParticleMaxSize(size);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setMaxRadius(size);
					}
				}
			}
		}
		
		public function setParticleMinSpin(spin:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setParticleMinSpin(spin);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setMinSpin(spin);
					}
				}
			}
		}
		
		public function setParticleMaxSpin(spin:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setParticleMaxSpin(spin);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setMaxSpin(spin);
					}
				}
			}
		}
		
		public function setSpringSpeed(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setSpringSpeed(val);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setSpringSpeed(val);
					}
				}
			}
		}
		
		public function setAtomicSpeed(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setAtomicSpeed(val);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setAtomicSpeed(val);
					}
				}
			}
		}
		
		public function setVisualMode(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setVisualMode(val);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setVisualMode(val);
					}
				}
			}
		}
		
		public function setGravityMode(val:Number){
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setGravityMode(val);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						//if(val > 1){
							values[i].die();
						//}
					}
				}
				if(val != 1){
					createNewParticles(people.getValue(selectedPerson).id, people.getValue(selectedPerson).x, people.getValue(selectedPerson).y);
				} else {
					createNewParticlesOutside(people.getValue(selectedPerson).id, people.getValue(selectedPerson).x, people.getValue(selectedPerson).y);
				}
			}
		}
		
		
		public function setSparkSpeed(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setSparkSpeed(val);
			}
		}
		
		public function setSparkLifeMin(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setSparkLifeMin(val);
			}
		}
		
		public function setSparkLifeMax(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setSparkLifeMax(val);
			}
		}
		
		public function setSparkEmitterDelay(val:Number):void{
			if(!isNaN(selectedPerson)){
				people.getValue(selectedPerson).setSparkEmitterDelay(val);
			}
		}
		
		public function setParticleCount(val:Number):void{
			
			if(!isNaN(selectedPerson)){
				var num:Number;
				if(val > people.getValue(selectedPerson).particleCount){
					num = Math.round(val - people.getValue(selectedPerson).particleCount);
					createNumParticles(num, selectedPerson, people.getValue(selectedPerson).x, people.getValue(selectedPerson).y);
				} else {
					num = Math.round(people.getValue(selectedPerson).particleCount - val);
					var removed:Number = 0;
					var values:Array = particles.getValues();
					for(var i:Number = 0; i<values.length; i++){
						if(values[i].emitterID == selectedPerson){
							values[i].die();
							removed++;
							if(removed >= num){
								break;
							}
						}
					}
				}
				people.getValue(selectedPerson).setParticleCount(val);
			}
		}
		
		public function keyDownListener(e:KeyboardEvent):void{
			if(e.keyCode == 187){	// create new person
				var xpos:Number = Math.random()*stage.stageWidth;
				var ypos:Number = (Math.random() * (stage.stageHeight - 150)) + 150;
				createPerson(xpos, ypos);
			} else if(e.keyCode == 189){ 
				var values:Array = people.getValues();
				if(values.length > 0){
					personLayer.removeChild(values[0]);
					people.remove(values[0].id);	// remove oldest person
				}
			} else {
				addPerson = true;
			}
		}
		
		public function keyUpListener(e:KeyboardEvent):void{
			addPerson = false;
		}
		
		public function mouseDownListener(event:MouseEvent):void{
			if(addPerson){
				createPerson(mouseX, mouseY);
			}
		}
		
		public function createPerson(xPos:Number, yPos:Number):void{
			var particleCount = 20;
			var radius:Number = 25;
			var mass:Number = 1;
			var torque:Number = (-0.1 * Math.random()) - 0.05;	// counter clockwise
			var person:Person = new Person(personID, xPos, yPos, radius, mass, torque);
			person.setParticleColor([Math.random()*255, Math.random()*255, Math.random()*255, 0.8]);
			personLayer.addChild(person);
			person.addCallback(this);
			people.put(personID, person);
			for(var p:Number = 0; p<particleCount; p++){
				createNewParticle(personID, xPos, yPos, -2 + (Math.random() * 4), 0);
			}
			personID++;
		}
		
	}
	
}