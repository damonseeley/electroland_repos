package net.electroland.kioskengine  {
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.text.TextFormatAlign;
	import flash.text.TextFieldAutoSize;
	import flash.display.BlendMode;
	import flash.events.*;
	import gs.*;
	
	public class TextItem extends Item{
		private var textLabel:TextField;
				
		public function TextItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
		}
		
		public function loadContent():void{
			textLabel = new TextField();
			var textFormat:TextFormat = new TextFormat();
			
			textLabel.wordWrap = true;
			textLabel.selectable = false;
			textLabel.blendMode = BlendMode.LAYER;
			if(itemEntries.hasOwnProperty("all")){
				textLabel.text = itemEntries["all"].getValue();
				textFormat.font = itemEntries["all"].font;
				textFormat.size = itemEntries["all"].size;
				textFormat.bold = itemEntries["all"].bold;
				textFormat.leading = itemEntries["all"].leading;
				if (itemEntries["all"].shadow) { createShadow() };
				textFormat.color = itemEntries["all"].color;
				if(itemEntries["all"].textalign == "left"){
					textFormat.align = TextFormatAlign.LEFT;
				} else if(itemEntries["all"].textalign == "right"){
					textFormat.align = TextFormatAlign.RIGHT;
				} else if(itemEntries["all"].textalign == "center"){
					textFormat.align = TextFormatAlign.CENTER;
				} else if(itemEntries["all"].textalign == "justify"){
					textFormat.align = TextFormatAlign.JUSTIFY;
				}
				if(itemEntries["all"].hasOwnProperty("x")){
					x = itemEntries["all"]["x"];
				}
				if(itemEntries["all"].hasOwnProperty("y")){
					y = itemEntries["all"]["y"];
				}
			} else {
				textLabel.text = itemEntries[conductor.getLanguage()].value;
				textFormat.font = itemEntries[conductor.getLanguage()].font;
				textFormat.size = itemEntries[conductor.getLanguage()].size;
				textFormat.bold = itemEntries[conductor.getLanguage()].bold;
				textFormat.leading = itemEntries[conductor.getLanguage()].leading;
					//trace("TEXT BOLDING IN TEXT ITEM: ", textFormat.bold);
				if (itemEntries[conductor.getLanguage()].shadow) { createShadow() };
				textFormat.color = itemEntries[conductor.getLanguage()].color;
				if(itemEntries[conductor.getLanguage()].textalign == "left"){
					textFormat.align = TextFormatAlign.LEFT;
				} else if(itemEntries[conductor.getLanguage()].textalign == "right"){
					textFormat.align = TextFormatAlign.RIGHT;
				} else if(itemEntries[conductor.getLanguage()].textalign == "center"){
					textFormat.align = TextFormatAlign.CENTER;
				} else if(itemEntries[conductor.getLanguage()].textalign == "justify"){
					textFormat.align = TextFormatAlign.JUSTIFY;
				}
				if(itemEntries[conductor.getLanguage()].hasOwnProperty("x")){
					x = itemEntries[conductor.getLanguage()]["x"];
				}
				if(itemEntries[conductor.getLanguage()].hasOwnProperty("y")){
					y = itemEntries[conductor.getLanguage()]["y"];
				}
			}
			textLabel.width = getWidth();
			//textLabel.height = getHeight();
			textLabel.autoSize = TextFieldAutoSize.LEFT;
			textLabel.setTextFormat(textFormat);
			
			/*
			// for diagnostics only
			textLabel.background = true;
			textLabel.backgroundColor = 0xFF0000;
			*/
			
			if(getHorizontalAlign() == "center"){
				textLabel.x = 0 - getWidth()/2;
			} else if(getHorizontalAlign() == "right"){
				textLabel.x = 0 - getWidth();
			}
			if(getVerticalAlign() == "center"){
				textLabel.y = 0 - textLabel.height/2;
			} else if(getVerticalAlign() == "bottom"){
				textLabel.y = 0 - textLabel.height;
			}
			addChild(textLabel);
			
			if(contentLoadedCallback != null){
				contentLoadedCallback(this);	// tell unit this has been loaded
			}
		}
		
		private function createShadow():void {
			trace("TextItem: Drawing Shadow");
			TweenMax.to(textLabel, 0.0, {dropShadowFilter:{color:0x000000, alpha:0.5, blurX:2, blurY:2, strength:1.0, distance:2}});
		}
		
		public function startTimer():void{
			// begin timers on various effects
			startEffectTimers();
		}
		
		override public function reset():void{
			removeChild(textLabel);
			super.reset();
		}
		
	}
	
}