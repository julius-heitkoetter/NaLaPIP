MODEL_TEMPLATE = """
var makeWorld = function(noise_pos, noise_size) {{
  
  var worldWidth = 600;
  var worldHeight = 500;

    //// Object definitions ////

    var ground = {{
      shape: 'rect',
      static: true,
      dims: [100000 * worldWidth, 5],
      x: worldWidth / 2,
      y: worldHeight - 5,
      type: "ground",
    }}

    var table = {{
      shape: 'rect',
      static: true,
      dims: [125, 82.5],
      x: 300,
      y: worldHeight-92.5,
      type: "table",
    }}
    
    var generate_boxes_from_arr = function(pos_arr, size_arr, color_arr, pos_noise, size_noise, boxes_so_far) {{
      var i = boxes_so_far.length;
      var box = {{
        shape: 'rect',
        static: false,
        dims: [size_arr[i][0] + gaussian({{mu: 0, sigma: size_noise}}), size_arr[i][1] + gaussian({{mu: 0, sigma: size_noise}})], 
        x: pos_arr[i][0] + gaussian({{mu: 0, sigma: pos_noise}}),
        y: pos_arr[i][1] + gaussian({{mu: 0, sigma: pos_noise}}),
        color: color_arr[i],
        type: "box",
      }}
      
      var boxes = boxes_so_far.concat([box])
      
      if (boxes.length == pos_arr.length) {{
        return boxes
      }} else {{
        return generate_boxes_from_arr(pos_arr, size_arr, color_arr, pos_noise, size_noise, boxes)
      }}
    }}
    
    {BOX_ENSEMBLE}
    
    var world = {{
      boxes: boxes,
      table: table,
      ground: ground,
    }} 
    
    return world
    
    }}
  ///////////////////
  //// Semantics ////
  ///////////////////

var isOnPlatform = function(obj) {{
  return obj.y < 325;
}}

var isNotOnPlatform = function(obj) {{
  return !(obj.y < 325);
}}

var isLeftOfPlatform = function(obj) {{
  return obj.y > 325 && obj.x < 175;
}}

var isNotLeftOfPlatform = function(obj) {{
  return !(obj.y > 325 && obj.x < 175);
}}

var isRightOfPlatform = function(obj) {{
  return obj.y > 325 && obj.x > 425;
}}

var isNotRightOfPlatform = function(obj) {{
  return !(obj.y > 325 && obj.x > 425);
}}

var isPurple = function(obj) {{
  return obj.color=="purple";
}}

var isRed = function(obj) {{
  return obj.color=="red";
}}

var isLarge = function(obj) {{
  return obj.dims[0] > 25 || obj.dims[1] > 25;
}}

var isNotLarge = function(obj) {{
  return !(obj.dims[0] > 25 || obj.dims[1] > 25);
}}

var flattenWorld = function (world) {{
    return [world.ground, world.table].concat(world.boxes);
  }}

var run = function (world) {{
  
  // Run World
  var flatWorld = flattenWorld(world);
  //physics.animate(1000, flatWorld);
  var finalWorld = physics.run(1000, flatWorld)
  
  var finalBoxes = finalWorld.slice(0,8);
  
  var conditional = {CONDITIONAL}
  return conditional ? 1 : 0;
}}

var result = function() {{
  var d = Infer({{method: 'rejection', samples: {N_SIMULATIONS}, incremental: true}},
               function() {{ return run(makeWorld( {PHYSICAL_GAUSSIAN_POS_NOISE}, {PHYSICAL_GAUSSIAN_POS_NOISE} )) }})
  var noProb = Math.exp(d.score(0))
  var yesProb = Math.exp(d.score(1))
  var likert = Math.round((yesProb / (noProb + yesProb)) * 6) + 1;
  return likert;
}}

var dist = Infer({{ method: 'forward', samples: {N_PARTICIPANTS} }}, result);

 var getData = function (key) {{
    return Math.exp(dist.score(key));
  }}
  var output = {{
    "probs": map(getData, dist.support()),
    "support": dist.support()
  }}

//var flatWorld = flattenWorld(makeWorld(0.0001));
//physics.animate(1000, flatWorld);
JSON.stringify(output);
"""