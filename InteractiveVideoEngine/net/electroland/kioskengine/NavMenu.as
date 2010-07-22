package net.electroland.kioskengine {
	
	import flash.display.Sprite;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.*;
	
	import net.electroland.utils.*;
	
	public class NavMenu extends Sprite{
		
		private var conductor:Conductor;
		private var textFormat:TextFormat;
		private var titleField:TextField;
		private var menuBG:Sprite;
		private var moduleButtons:Array;		// stores roll-over triggers to display units
		
		/*
		
		NAVMENU.as
		by Aaron Siegel, 9-1-09
		
		Nested drop down list of modules and units for test navigation of the content.
		
		*/
		
		public function NavMenu(){
			textFormat = new TextFormat();
			textFormat.font = "Arial";
			textFormat.size = 10;
			titleField = new TextField();
			titleField.text = "MODULES:";
			titleField.selectable = false;
			titleField.textColor = 0x000000;
			titleField.blendMode = BlendMode.LAYER;
			titleField.background = true;
			titleField.autoSize = TextFieldAutoSize.LEFT;
			titleField.setTextFormat(textFormat);
			
			addChild(titleField);
			
			this.addEventListener(MouseEvent.MOUSE_OVER, mainMouseOver);
			this.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
		}
		
		public function buildMenu(conductor:Conductor){
			this.conductor = conductor;
			menuBG = new Sprite();
			var itemOffset:Number = 14;
			moduleButtons = new Array();
			var moduleNames:Array = conductor.getModuleNames();
			//trace("NUMBER OF MODULES: "+moduleNames.length);
			for (var i=0; i<moduleNames.length; i++){
				var menuItem:NavMenuItem = new NavMenuItem(conductor, moduleNames[i], textFormat);
				menuItem.y = itemOffset*i;
				moduleButtons.push(menuItem);
				//trace(moduleNames[i]+" added to the nav menu");
				menuBG.addChild(menuItem);
			}
			
			menuBG.y = itemOffset;
			menuBG.visible = false;
			menuBG.addChildAt(ELGfx.filledRect(0,0,100,menuBG.height,0xFFFFFF,0), 0);
			menuBG.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			this.addChild(menuBG);
			
			for each (var modButton:NavMenuItem in moduleButtons){
				modButton.buildMenu();
			}
			
		}
		
		private function addItemEvents():void {
			this.addEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			for each (var i:NavMenuItem in moduleButtons){
				//i.addEventListener(MouseEvent.MOUSE_UP, mousePressed);
				//for each (var unitButton:Sprite in i.unitList){
					//unitButton.addEventListener(MouseEvent.MOUSE_UP, mousePressed);
				//}
			}
		}
		
		private function removeItemEvents():void {
			this.removeEventListener(MouseEvent.MOUSE_OUT, mainMouseOut);
			for each (var i:NavMenuItem in moduleButtons){
				//i.removeEventListener(MouseEvent.MOUSE_UP, mousePressed);
			}
		}
		
		private function hideMenu():void {
			removeItemEvents();
			menuBG.visible = false;
		}
		
		private function showMenu():void {
			addItemEvents();
			menuBG.visible = true;
		}
		
		private function mainMouseOver(event:MouseEvent):void{
			showMenu();
			if (event.target is TextField) {
				var tf:TextFormat = event.target.getTextFormat();
				tf.bold = true;
				event.target.setTextFormat(tf);
				event.target.textColor = 0xaa0000;
				// DISPLAY UNITS HERE
				if(event.target.parent is NavMenuItem){
					for each(var menuItem:NavMenuItem in moduleButtons){
						if(menuItem != event.target.parent){
							menuItem.hideMenu();
						}
					}
					event.target.parent.showMenu();
				}
			}
		}
		
		private function mainMouseOut(event:MouseEvent):void{
			hideMenu();
			if (event.target is TextField) {
				var tf:TextFormat = event.target.getTextFormat();
				tf.bold = false;
				event.target.setTextFormat(tf);
				event.target.textColor = 0x000000;
				// HIDE UNITS HERE
				if(event.target.parent is NavMenuItem){
					//event.target.parent.hideMenu();
				}
			}
		}
		
	}
	
}