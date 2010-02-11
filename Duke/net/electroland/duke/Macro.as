package net.electroland.duke {
	
	public class Macro{
		
		public var id:Number;	// number that activates macro
		public var attractionRadiusMax:Number;
		public var attractionRadiusMin:Number;
		public var repulsionRadius:Number;
		public var mass:Number;
		public var torque:Number;
		public var red:Number;
		public var green:Number;
		public var blue:Number;
		public var minRadius:Number;
		public var maxRadius:Number;
		public var minSpin:Number;
		public var maxSpin:Number;
		public var visualMode:Number;
		public var gravityMode:Number;
		
		public function Macro(id:Number){
			this.id = id;
		}
		
		public function setValues(attractionRadiusMin:Number, attractionRadiusMax:Number, repulsionRadius:Number,
									 mass:Number, torque:Number, red:Number, green:Number, blue:Number, minRadius:Number,
									 maxRadius:Number, minSpin:Number, maxSpin:Number, visualMode:Number, gravityMode:Number):void{
			this.attractionRadiusMin = attractionRadiusMin;
			this.attractionRadiusMax = attractionRadiusMax;
			this.repulsionRadius = repulsionRadius;
			this.mass = mass;
			this.torque = torque;
			this.red = red;
			this.green = green;
			this.blue = blue;
			this.minRadius = minRadius;
			this.maxRadius = maxRadius;
			this.minSpin = minSpin;
			this.maxSpin = maxSpin;
			this.visualMode = visualMode;
			this.gravityMode = gravityMode;
		}
		
		public function loadValues(xml:XML){
			this.attractionRadiusMin = Number(xml.child("attractionRadiusMin").valueOf());
			this.attractionRadiusMax = Number(xml.child("attractionRadiusMax").valueOf());
			this.repulsionRadius = Number(xml.child("repulsionRadius").valueOf());
			this.mass = Number(xml.child("mass").valueOf());
			this.torque = Number(xml.child("torque").valueOf());
			this.red = Number(xml.child("red").valueOf());
			this.green = Number(xml.child("green").valueOf());
			this.blue = Number(xml.child("blue").valueOf());
			this.minRadius = Number(xml.child("minRadius").valueOf());
			this.maxRadius = Number(xml.child("maxRadius").valueOf());
			this.minSpin = Number(xml.child("minSpin").valueOf());
			this.maxSpin = Number(xml.child("maxSpin").valueOf());
			this.visualMode = Number(xml.child("visualMode").valueOf());
			this.gravityMode = Number(xml.child("gravityMode").valueOf());
		}
		
	}
	
}