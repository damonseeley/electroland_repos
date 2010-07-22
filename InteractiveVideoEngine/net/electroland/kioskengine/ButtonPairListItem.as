package net.electroland.kioskengine {
	
	public class ButtonPairListItem extends Item {
		private var buttonPairList:Array = new Array();
		private var submitButton:ButtonItem;
		private var submit:Boolean = false;
		private var onCompleteDispatched:Boolean = false;
		private var itemsLoaded = 0;
		
		/*
		
		BUTTONPAIRLISTITEM.as
		by Aaron Siegel, 7-8-09
		
		Contains an array of ButtonPairs which will return a true/false value depending on which
		button has been selected. ButtonPairList uses a "submit" button with an action string to
		tell it which analysis function to use against the pairs (ie: isAllOn, isAnyOn, isAllOff,
		etc).
		
		*/
		
		public function ButtonPairListItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
		}
		
		public function itemEventHandler(event:ItemEvent):void{
			// log the option picked and add to the answers array
			//trace(event.getAction()+" "+event.getArguments());
			var success:Boolean = true;
			var allActivated:Boolean = true;
			// make sure all button pairs have been activated first
			for each(var buttonPair in buttonPairList){
				//trace(buttonPair.isActivated());
				if(!buttonPair.isActivated()){
					allActivated = false;
				}
			}
			
			if(event.getAction() == "isAllFalse"){
				// check to see if every pair is set to false
				for each(buttonPair in buttonPairList){
					trace(buttonPair.getState());
					if(buttonPair.getState()){
						success = false;
						//break;
					}
				}
				submit = true;
			} else if(event.getAction() == "isAnyFalse"){
				success = false;
				for each(buttonPair in buttonPairList){
					if(!buttonPair.getState()){
						success = true;
						break;
					}
				}
				submit = true;
			} else if(event.getAction() == "isAllTrue"){
				for each(buttonPair in buttonPairList){
					if(!buttonPair.getState()){
						success = false;
						break;
					}
				}
				submit = true;
			} else if(event.getAction() == "isAnyTrue"){
				success = false;
				for each(buttonPair in buttonPairList){
					if(buttonPair.getState()){
						success = true;
						break;
					}
				}
				submit = true;
			} else if(event.getAction() == "option"){
				success = false;
				submit = false;
			}
			
			if(allActivated){	// if each pair has been answered...
				if(submit){
					if(success){	// if boolean combination fulfills submit button type
						var args:Array = event.getArguments().split(":");
						dispatchEvent(new ItemEvent(args[0], args[1]));
					} else {
						dispatchEvent(new ItemEvent("default", "default"));
					}
				} else {
					if(!onCompleteDispatched){
						submitButton.triggerOnCompleteEffects();
						dispatchEvent(new ItemEvent("onComplete", "default"));
						onCompleteDispatched = true;
					}
				}
			}
		}
		
		public function itemLoaded(item:Item):void{
			trace(getName()+".itemLoaded(): "+ item.getName() +" LOADED");
			itemsLoaded++;
			if(contentLoadedCallback != null){			// make sure there is a listener established
				if(itemsLoaded == buttonPairList.length + 1){	// inform the button list this pair (and submit button) has been fully loaded	
					contentLoadedCallback(this);
				}
			}
		}
		
		public function loadContent():void{
			for each (var buttonPair in buttonPairList){
				//trace(buttonPair.buttonList.length);
				buttonPair.addContentListener(itemLoaded);
				buttonPair.loadContent();
			}
			submitButton.addContentListener(itemLoaded);
			submitButton.loadContent();
		}
		
		override protected function parseXML(itemElement:XML):void{
			setID(itemElement.attribute("ID"));
			setName(itemElement.attribute("Name"));
			setType(itemElement.attribute("Type"));
			setDescription(itemElement.attribute("Description"));
			setLayer(Number(itemElement.attribute("Layer")));
			x = Number(itemElement.attribute("x"));				// x/y/width/height never used here
			y = Number(itemElement.attribute("y"));
			setWidth(Number(itemElement.attribute("width")));
			setHeight(Number(itemElement.attribute("height")));
			for each (var itemEventElement:XML in itemElement.ItemEvent){
				//trace(itemEventElement.attribute("Action"));
				parseEffects(itemEventElement);
			}
			for each (var nestedItemElement:XML in itemElement.Item){
				//trace(nestedItemElement);
				if(nestedItemElement.attribute("Type") == "ButtonPair"){
					var buttonPair:ButtonPairItem = new ButtonPairItem(conductor, nestedItemElement);
					buttonPairList.push(buttonPair);
				} else {
					submitButton = new ButtonItem(conductor, nestedItemElement);
				}
			}
			for each (var itemEntryElement:XML in itemElement.ItemEntry){
				parseItemEntries(itemEntryElement);
			}
		}
		
		override public function reset():void{
			// buttons have independent timers
			for each (var buttonPair in buttonPairList){
				removeChild(buttonPair);
				buttonPair.reset();
				buttonPair.removeEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			}
			removeChild(submitButton);
			submitButton.reset();
			submitButton.removeEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			super.reset();
			onCompleteDispatched = false;
			itemsLoaded = 0;
			submit = false;
		}
		
		public function startTimer():void{
			// buttons have independent timers
			for each (var buttonPair in buttonPairList){
				//trace(buttonPair.buttonList.length);
				addChild(buttonPair);
				buttonPair.startTimer();
				buttonPair.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			}
			addChild(submitButton);
			submitButton.startTimer();
			submitButton.addEventListener(ItemEvent.ITEM_EVENT, itemEventHandler);
			startEffectTimers();
		}
		
	}
	
}