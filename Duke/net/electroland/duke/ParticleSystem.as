package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.Event;
	import flash.display.Loader;
	import flash.display.LoaderInfo;
	import flash.net.URLRequest;
	import com.ericfeminella.collections.HashMap;
	
	public class ParticleSystem extends MovieClip{
		
		private var people:Array;				// container of person objects
		private var particles:HashMap;			// container of particle objects
		private var personID:Number;			// unique ID for each person
		private var particleID:Number;			// unique ID for each particle
		private var particleLayer:MovieClip;
		private var personLayer:MovieClip;
		private var colors:Array;				// preset colors
		private var selectedPerson:Number;		// id of selected person
		private var controlPanel:ControlPanel;	// reference just for updating values
		public var softParticle:Loader;			// images for particles
		public var weirdParticle:Loader;
		public var lineParticle:Loader;
		
		/*
		PARTICLESYSTEM.as
		by Aaron Siegel, 2-1-2010
		
		Controls updating the state of the system as a whole, including adding or
		removing Person objects or Particle objects.
		*/
		
		public function ParticleSystem(){
			particleLayer = new MovieClip();
			personLayer = new MovieClip();
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

			softParticle = new Loader();
			softParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			softParticle.load(new URLRequest("images/softParticle.png"));
			weirdParticle = new Loader();
			weirdParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			weirdParticle.load(new URLRequest("images/weirdParticle.png"));
			lineParticle = new Loader();
			lineParticle.contentLoaderInfo.addEventListener(Event.COMPLETE, imageLoaded);
			lineParticle.load(new URLRequest("images/lineParticle.png"));
			
			people = new Array();
			particles = new HashMap();
			for(var i:Number = 0; i<personCount; i++){
				var xPos:Number = Math.random()*stage.stageWidth;
				var yPos:Number = Math.random()*stage.stageHeight;
				var radius:Number = 25;
				var mass:Number = 1;
				var torque:Number = (-0.1 * Math.random()) - 0.05;	// counter clockwise
				var person:Person = new Person(personID, xPos, yPos, radius, mass, torque);
				person.setParticleColor(colors[i]);
				person.addCallback(this);
				personLayer.addChild(person);
				people.push(person);
				for(var p:Number = 0; p<particleCount; p++){
					createNewParticle(personID, xPos, yPos, 3 + (Math.random() * 3), -2 + (Math.random() * 4), 0);
				}
				personID++;
			}
			
			this.addEventListener(Event.ENTER_FRAME, update);
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
				values[i].applyRotation(people);
			}
			for(i = 0; i<values.length; i++){
				values[i].applySquareRotation(people);
			}
			for(i = 0; i<values.length; i++){
				values[i].applyGravity(people);
			}
			for(i = 0; i<values.length; i++){
				values[i].applySquareGravity(people);
			}
			for(i = 0; i<values.length; i++){
				values[i].applyStarGravity(people);
			}
		}
		
		public function createNewParticle(emitterID:Number, xPos:Number, yPos:Number, radius:Number, spin:Number, visualMode:Number):void{
			// emit particles from this point with an initial random vector
			var mass:Number = Math.random() + 0.1;	// 0.1 - 1
			var particle:Particle = new Particle(particleID, emitterID, xPos, yPos, radius, mass, spin, visualMode, this);
			particle.setColor(people[emitterID].getParticleColor());
			particleLayer.addChild(particle);
			particles.put(particleID, particle);
			particleID++;
		}
		
		public function removeParticle(e:ParticleEvent){
			particles.remove(e.id);	// remove particle
			//trace("particle "+ e.id +" removed, "+ particles.size() + " left");
		}
		
		public function personSelected(id:Number){
			selectedPerson = id;
			for(var i:Number = 0; i<people.length; i++){
				if(people[i].id != id){
					people[i].deselect();
				}
			}
			// update control panel
			controlPanel.updateValues(people[id].radiusOfAttractionMin, people[id].radiusOfAttractionMax, people[id].radiusOfRepulsion,
									  people[id].mass, people[id].torque, people[id].particleColorRed, people[id].particleColorGreen,
									  people[id].particleColorBlue, people[id].particleMinRadius, people[id].particleMaxRadius,
									  people[id].particleSpinMin, people[id].particleSpinMax, people[id].visualMode, people[id].gravityMode);
		}
		
		
		
		// FUNCTIONS FOR MODIFYING PERSON AND PROPERTY VALUES ON THE FLY
		// TODO: needs to be changed to support people hashmap
		
		public function addControlPanel(controlPanel:ControlPanel):void{
			this.controlPanel = controlPanel;
		}
		
		public function setRadiusOfAttractionMax(radiusOfAttractionMax:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setRadiusOfAttractionMax(radiusOfAttractionMax);
			}
		}
		
		public function setRadiusOfAttractionMin(radiusOfAttractionMin:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setRadiusOfAttractionMin(radiusOfAttractionMin);
			}
		}
		
		public function setRadiusOfRepulsion(radiusOfRepulsion:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setRadiusOfRepulsion(radiusOfRepulsion);
			}
		} 
		
		public function setTorque(torque:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setTorque(torque);
			}
		}
		
		public function setMass(mass:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setMass(mass);
			}
		}
		
		public function setRed(red:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setRed(red);
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
				people[selectedPerson].setGreen(green);
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
				people[selectedPerson].setBlue(blue);
				var values:Array = particles.getValues();
				for(var i:Number = 0; i<values.length; i++){
					if(values[i].emitterID == selectedPerson){
						values[i].setBlue(blue);
					}
				}
			}
		}
		
		public function setParticleMinSize(size:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setParticleMinSize(size);
			}
		}
		
		public function setParticleMaxSize(size:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setParticleMaxSize(size);
			}
		}
		
		public function setParticleMinSpin(spin:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setParticleMinSpin(spin);
			}
		}
		
		public function setParticleMaxSpin(spin:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setParticleMaxSpin(spin);
			}
		}
		
		public function setVisualMode(val:Number):void{
			if(!isNaN(selectedPerson)){
				people[selectedPerson].setVisualMode(val);
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
				people[selectedPerson].setGravityMode(val);
			}
		}
		
	}
	
}