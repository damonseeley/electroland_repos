﻿package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.Event;
	import flash.events.KeyboardEvent;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	import com.ericfeminella.collections.HashMap;
	
	public class MacroManager extends MovieClip{
		
		private var particleSystem:ParticleSystem;	// references needed to update values
		private var controlPanel:ControlPanel;
		private var macros:HashMap;					// store macro objects in a hashmap for lookup purposes
		private var xmlLoader:URLLoader  = new URLLoader();
		private var xmlData:XML = new XML();
		
		/*
		
		MACROMANAGER.as
		by Aaron Siegel, 2-9-2010
		
		(1) Stores values of properties from the control panel in macros assigned to number keys.
		(2) Print out all macros in XML format to console.
		(3) Load all macros from start in XML format.
		(4) Restore values to control panel and selected person when macro is pressed.
		
		*/
		
		public function MacroManager(particleSystem:ParticleSystem, controlPanel:ControlPanel){
			this.particleSystem = particleSystem;
			this.controlPanel = controlPanel;
		}
		
		public function start():void{
			macros = new HashMap();
			stage.addEventListener(KeyboardEvent.KEY_DOWN,keyDownListener);
			//xmlLoader.addEventListener(Event.COMPLETE, parseXML);
			//xmlLoader.load(new URLRequest("macros.xml")); 
			
			//var macroOne = new Macro(1);
			//macroOne.setValues(100, 300, 50, 1, -0.25, 200, 150, 0, 2, 25, -1, 1, 5, 4, 2, 1, 60, 3, 2, 5, 50);
			//macros.put(1, macroOne);
			
			var macroOne = new Macro(1);
			var macroTwo = new Macro(2);
			var macroThree = new Macro(3);
			var macroFour = new Macro(4);
			var macroFive = new Macro(5);
			var macroSix = new Macro(6);
			macroOne.setValues(100, 300, 50, 1, -0.25, 200, 150, 0, 2, 25, -1, 1, 5, 4, 2.08, 1, 80, 3, 2, 5, 50);
			macroTwo.setValues(86.75, 300, 27, 0.6, 0.879, 200, 150, 0, 2, 14.72, -1, 2, 1, 0, 2, 1, 70, 3, 2, 5, 50);
			macroThree.setValues(181.75, 300, 50, 1.85, 0.2, 0, 255, 117, 2, 19.13, -1, 2.6, 4, 3, 2.08, 3.961, 75, 3, 2, 4.755, 50);
			macroFour.setValues(181.75, 300, 50, 1.85, 0.2, 0, 255, 117, 2, 15.7, -1, 2.6, 3, 5, 2.08, 3.367, 75, 8.6569, 1.962, 2.158, 10.99);
			macroFive.setValues(181.75, 300, 50, 1.95, 0.2, 0, 255, 117, 2, 15.7, -1, 2.6, 6, 3, 2.08, 2.377, 88, 8.6569, 1.325, 1.521, 50.95);
			macroSix.setValues(181.75, 305.25, 40, 0.35, -0.196, 0, 87, 255, 2, 8.84, -14.2, 14, 2, 4, 2.971, 1, 100, 3, 2, 3.138, 50);
			macros.put(1, macroOne);
			macros.put(2, macroTwo);
			macros.put(3, macroThree);
			macros.put(4, macroFour);
			macros.put(5, macroFive);
			macros.put(6, macroSix);
			
			applyMacro(1);
		}
		
		function keyDownListener(e:KeyboardEvent) {
			if(e.keyCode - 48 >= 0 && e.keyCode - 48 < 10){
				if(e.shiftKey){	// save
					saveMacro(e.keyCode - 48);
				} else {
					applyMacro(e.keyCode - 48);
				}
			} else if(e.keyCode == 186) {	// semi-colon
				//printXML();
				printValues();
			}
		}
		
		public function applyMacro(id:Number):void{
			// a macro key is pressed, and the stored values are applied to the selected person and control panel.
			if(macros.containsKey(id)){
				controlPanel.loadMacro(macros.getValue(id));
				trace(id +" applied");
			}
		}
		
		public function saveMacro(id:Number):void{
			// a macro key is pressed while holding shift, and the values from the control panel are stored to a macro object.
			var macro:Macro = controlPanel.getMacro(id);
			macros.put(id, macro);
			trace(id +" saved");
		}
		
		public function parseXML(e:Event):void{
			xmlData = new XML(e.target.data);
			for each (var macroElement:XML in xmlData.children()){
				var id:Number = Number(macroElement.child("id").valueOf());
				var macro:Macro = new Macro(id);
				macro.loadValues(macroElement);
				macros.put(id, macro);
				//trace(macros.size());
			}
		}
		
		public function printValues():void{
			var values:Array = macros.getValues();
			for(var i=0; i<values.length; i++){
				trace(values[i].id +": "+ values[i].attractionRadiusMin +", "+ values[i].attractionRadiusMax +", "+ values[i].repulsionRadius +", "+ values[i].mass +", "+ values[i].torque +", "+ values[i].red +", "+ values[i].green +", "+ values[i].blue +", "+ values[i].minRadius +", "+ values[i].maxRadius +", "+ values[i].minSpin +", "+ values[i].maxSpin +", "+ values[i].visualMode +", "+ values[i].gravityMode +", "+ values[i].atomicSpeed +", "+ values[i].springSpeed +", "+ values[i].particleCount +", "+ values[i].sparkSpeed +", "+ values[i].sparkLifeMin +", "+ values[i].sparkLifeMax +", "+ values[i].emitterDelay);
			}
		}
		
		public function printXML():void{
			// print out all macros in XML format
			trace("<Macros>");
			var values:Array = macros.getValues();
			for(var i=0; i<values.length; i++){
				trace("<Macro>");
				
				trace("<id>"+ values[i].id + "</id>");
				trace("<attractionRadiusMin>" + values[i].attractionRadiusMin + "</attractionRadiusMin>");
				trace("<attractionRadiusMax>" + values[i].attractionRadiusMax +"</attractionRadiusMax>");
				trace("<repulsionRadius>" + values[i].repulsionRadius + "</repulsionRadius>");
				trace("<mass>" + values[i].mass + "</mass>");
				trace("<torque>" + values[i].torque + "</torque>");
				trace("<red>" + values[i].red + "</red>");
				trace("<green>" + values[i].green + "</green>");
				trace("<blue>" + values[i].blue + "</blue>");
				trace("<minRadius>" + values[i].minRadius + "</minRadius>");
				trace("<maxRadius>" + values[i].maxRadius + "</maxRadius>");
				trace("<minSpin>" + values[i].minSpin + "</minSpin>");
				trace("<maxSpin>" + values[i].maxSpin + "</maxSpin>");
				trace("<visualMode>" + values[i].visualMode + "</visualMode>");
				trace("<gravityMode>" + values[i].gravityMode + "</gravityMode>");
				trace("<atomicSpeed>" + values[i].atomicSpeed + "</atomicSpeed>");
				trace("<springSpeed>" + values[i].springSpeed + "</springSpeed>");
				trace("<particleCount>" + values[i].particleCount + "</particleCount>");
				trace("<sparkSpeed>" + values[i].sparkSpeed + "</sparkSpeed>");
				trace("<sparkLifeMin>" + values[i].sparkLifeMin + "</sparkLifeMin>");
				trace("<sparkLifeMax>" + values[i].sparkLifeMax + "</sparkLifeMax>");
				trace("<emitterDelay>" + values[i].emitterDelay + "</emitterDelay>");
				
				trace("</Macro>");
			}
			trace("</Macros>");
		}
		
	}
	
}