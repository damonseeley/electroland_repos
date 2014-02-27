{
	"patcher" : 	{
		"fileversion" : 1,
		"rect" : [ 880.0, 416.0, 556.0, 424.0 ],
		"bglocked" : 0,
		"defrect" : [ 880.0, 416.0, 556.0, 424.0 ],
		"openrect" : [ 0.0, 0.0, 0.0, 0.0 ],
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 0,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 0,
		"toolbarvisible" : 1,
		"boxanimatetime" : 200,
		"imprint" : 0,
		"metadata" : [  ],
		"boxes" : [ 			{
				"box" : 				{
					"maxclass" : "inlet",
					"id" : "obj-1",
					"numinlets" : 0,
					"numoutlets" : 1,
					"patching_rect" : [ 746.0, 252.0, 25.0, 25.0 ],
					"outlettype" : [ "" ],
					"hidden" : 1,
					"comment" : ""
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "button",
					"id" : "obj-189",
					"numinlets" : 1,
					"numoutlets" : 1,
					"patching_rect" : [ 542.0, 296.0, 20.0, 20.0 ],
					"outlettype" : [ "bang" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "message",
					"text" : "default vol",
					"fontname" : "Arial",
					"id" : "obj-187",
					"numinlets" : 2,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 476.0, 266.0, 64.0, 18.0 ],
					"outlettype" : [ "" ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "loadmess 100",
					"fontname" : "Arial",
					"id" : "obj-185",
					"numinlets" : 1,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 542.0, 318.0, 102.0, 20.0 ],
					"outlettype" : [ "" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "pow 2.",
					"fontname" : "Arial",
					"id" : "obj-182",
					"numinlets" : 2,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 650.0, 221.0, 46.0, 20.0 ],
					"outlettype" : [ "float" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "* 0.01",
					"fontname" : "Arial",
					"id" : "obj-181",
					"numinlets" : 2,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 650.0, 199.0, 42.0, 20.0 ],
					"outlettype" : [ "float" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "slider",
					"id" : "obj-179",
					"numinlets" : 1,
					"size" : 110.0,
					"numoutlets" : 1,
					"patching_rect" : [ 491.0, 120.0, 27.0, 141.0 ],
					"outlettype" : [ "" ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "send~ mFader",
					"fontname" : "Arial",
					"id" : "obj-176",
					"numinlets" : 1,
					"fontsize" : 12.0,
					"numoutlets" : 0,
					"patching_rect" : [ 650.0, 287.0, 88.0, 20.0 ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "slide~ 10. 10.",
					"fontname" : "Arial",
					"id" : "obj-173",
					"numinlets" : 3,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 650.0, 265.0, 82.0, 20.0 ],
					"outlettype" : [ "signal" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "sig~",
					"fontname" : "Arial",
					"id" : "obj-172",
					"numinlets" : 1,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 650.0, 243.0, 33.0, 20.0 ],
					"outlettype" : [ "signal" ],
					"hidden" : 1
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "thispatcher",
					"fontname" : "Arial",
					"id" : "obj-71",
					"numinlets" : 1,
					"fontsize" : 12.0,
					"numoutlets" : 2,
					"patching_rect" : [ 651.0, 341.0, 69.0, 20.0 ],
					"outlettype" : [ "", "" ],
					"hidden" : 1,
					"save" : [ "#N", "thispatcher", ";", "#Q", "end", ";" ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "newobj",
					"text" : "p generatemeters",
					"fontname" : "Arial",
					"id" : "obj-70",
					"numinlets" : 0,
					"fontsize" : 12.0,
					"numoutlets" : 1,
					"patching_rect" : [ 651.0, 315.0, 105.0, 20.0 ],
					"outlettype" : [ "" ],
					"hidden" : 1,
					"patcher" : 					{
						"fileversion" : 1,
						"rect" : [ 50.0, 94.0, 640.0, 480.0 ],
						"bglocked" : 0,
						"defrect" : [ 50.0, 94.0, 640.0, 480.0 ],
						"openrect" : [ 0.0, 0.0, 0.0, 0.0 ],
						"openinpresentation" : 0,
						"default_fontsize" : 12.0,
						"default_fontface" : 0,
						"default_fontname" : "Arial",
						"gridonopen" : 0,
						"gridsize" : [ 15.0, 15.0 ],
						"gridsnaponopen" : 0,
						"toolbarvisible" : 1,
						"boxanimatetime" : 200,
						"imprint" : 0,
						"metadata" : [  ],
						"boxes" : [ 							{
								"box" : 								{
									"maxclass" : "comment",
									"text" : "also this outlet has been disconnected to avoid accidentally re-generating them\n",
									"linecount" : 2,
									"fontname" : "Arial",
									"id" : "obj-3",
									"numinlets" : 1,
									"fontsize" : 12.0,
									"numoutlets" : 0,
									"patching_rect" : [ 112.0, 239.0, 269.0, 34.0 ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "comment",
									"text" : "this is only for reference or re-generating the meters. to remove original meters delete them manually",
									"linecount" : 2,
									"fontname" : "Arial",
									"id" : "obj-2",
									"numinlets" : 1,
									"fontsize" : 12.0,
									"numoutlets" : 0,
									"patching_rect" : [ 163.0, 27.0, 305.0, 34.0 ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "button",
									"id" : "obj-21",
									"numinlets" : 1,
									"numoutlets" : 1,
									"patching_rect" : [ 74.0, 18.0, 20.0, 20.0 ],
									"outlettype" : [ "bang" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "newobj",
									"text" : "t i i",
									"fontname" : "Arial",
									"id" : "obj-20",
									"numinlets" : 1,
									"fontsize" : 12.0,
									"numoutlets" : 2,
									"patching_rect" : [ 93.0, 81.0, 32.5, 20.0 ],
									"outlettype" : [ "int", "int" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "newobj",
									"text" : "pack 0 0",
									"fontname" : "Arial",
									"id" : "obj-19",
									"numinlets" : 2,
									"fontsize" : 12.0,
									"numoutlets" : 1,
									"patching_rect" : [ 102.0, 167.0, 56.0, 20.0 ],
									"outlettype" : [ "" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "newobj",
									"text" : "* 40",
									"fontname" : "Arial",
									"id" : "obj-16",
									"numinlets" : 2,
									"fontsize" : 12.0,
									"numoutlets" : 1,
									"patching_rect" : [ 139.0, 140.0, 32.5, 20.0 ],
									"outlettype" : [ "int" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "newobj",
									"text" : "- 1",
									"fontname" : "Arial",
									"id" : "obj-15",
									"numinlets" : 2,
									"fontsize" : 12.0,
									"numoutlets" : 1,
									"patching_rect" : [ 139.0, 116.0, 32.5, 20.0 ],
									"outlettype" : [ "int" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "newobj",
									"text" : "uzi 24",
									"fontname" : "Arial",
									"id" : "obj-14",
									"numinlets" : 2,
									"fontsize" : 12.0,
									"numoutlets" : 3,
									"patching_rect" : [ 79.0, 49.0, 46.0, 20.0 ],
									"outlettype" : [ "bang", "bang", "int" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "message",
									"text" : "script newobject bpatcher @name meter.maxpat @args $1 @patching_rect $2 0 42 196",
									"fontname" : "Arial",
									"id" : "obj-18",
									"numinlets" : 2,
									"fontsize" : 12.0,
									"numoutlets" : 1,
									"patching_rect" : [ 73.0, 210.0, 543.0, 18.0 ],
									"outlettype" : [ "" ]
								}

							}
, 							{
								"box" : 								{
									"maxclass" : "outlet",
									"id" : "obj-17",
									"numinlets" : 1,
									"numoutlets" : 0,
									"patching_rect" : [ 74.0, 259.0, 25.0, 25.0 ],
									"comment" : ""
								}

							}
 ],
						"lines" : [ 							{
								"patchline" : 								{
									"source" : [ "obj-21", 0 ],
									"destination" : [ "obj-14", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-14", 2 ],
									"destination" : [ "obj-20", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-19", 0 ],
									"destination" : [ "obj-18", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-20", 0 ],
									"destination" : [ "obj-19", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-20", 1 ],
									"destination" : [ "obj-15", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-16", 0 ],
									"destination" : [ "obj-19", 1 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
, 							{
								"patchline" : 								{
									"source" : [ "obj-15", 0 ],
									"destination" : [ "obj-16", 0 ],
									"hidden" : 0,
									"midpoints" : [  ]
								}

							}
 ]
					}
,
					"saved_object_attributes" : 					{
						"default_fontsize" : 12.0,
						"fontname" : "Arial",
						"fontface" : 0,
						"default_fontface" : 0,
						"fontsize" : 12.0,
						"globalpatchername" : "",
						"default_fontname" : "Arial"
					}

				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 1 ],
					"id" : "obj-125",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 0.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 2 ],
					"id" : "obj-127",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 39.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 3 ],
					"id" : "obj-129",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 78.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 4 ],
					"id" : "obj-131",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 117.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 5 ],
					"id" : "obj-133",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 156.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 6 ],
					"id" : "obj-135",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 195.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 7 ],
					"id" : "obj-137",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 234.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 8 ],
					"id" : "obj-139",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 273.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 9 ],
					"id" : "obj-141",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 312.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 10 ],
					"id" : "obj-143",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 351.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 11 ],
					"id" : "obj-145",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 390.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 12 ],
					"id" : "obj-147",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 429.0, 0.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 13 ],
					"id" : "obj-149",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 0.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 14 ],
					"id" : "obj-151",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 39.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 15 ],
					"id" : "obj-153",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 78.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 16 ],
					"id" : "obj-155",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 117.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 17 ],
					"id" : "obj-157",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 156.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 18 ],
					"id" : "obj-159",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 195.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 19 ],
					"id" : "obj-161",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 234.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 20 ],
					"id" : "obj-163",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 273.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 21 ],
					"id" : "obj-165",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 312.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 22 ],
					"id" : "obj-167",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 351.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 23 ],
					"id" : "obj-169",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 390.0, 210.0, 42.0, 196.0 ]
				}

			}
, 			{
				"box" : 				{
					"maxclass" : "bpatcher",
					"args" : [ 24 ],
					"id" : "obj-171",
					"numinlets" : 0,
					"numoutlets" : 0,
					"name" : "meter.maxpat",
					"patching_rect" : [ 429.0, 210.0, 42.0, 196.0 ]
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"source" : [ "obj-70", 0 ],
					"destination" : [ "obj-71", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-172", 0 ],
					"destination" : [ "obj-173", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-173", 0 ],
					"destination" : [ "obj-176", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-181", 0 ],
					"destination" : [ "obj-182", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-179", 0 ],
					"destination" : [ "obj-181", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-182", 0 ],
					"destination" : [ "obj-172", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-185", 0 ],
					"destination" : [ "obj-179", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-189", 0 ],
					"destination" : [ "obj-185", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
, 			{
				"patchline" : 				{
					"source" : [ "obj-187", 0 ],
					"destination" : [ "obj-189", 0 ],
					"hidden" : 1,
					"midpoints" : [  ]
				}

			}
 ]
	}

}
