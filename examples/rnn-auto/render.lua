require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
json = require 'json'

dofile("sample.lua")
local app = require('../../waffle')

app.get('/', function(req, res)
    res.render('main.html')
    print(output)
end)

app.ws('/ws', function(ws)
   ws.checkorigin = function(origin)
      print(origin)
      return origin == 'http://deeplearn2.eecs.umich.edu:8080'
   end

   ws.onopen = function(req)
      print('/ws/opened')
      --ws:write('Yo')
   end

   ws.onmessage = function(data)
      local seed_text = data.data
      if string.len(seed_text) > 0 then
	      output = sample(seed_text, {})
	  else
	      output = ""
	  end
      ws:write(json:encode(output))
      --ws:close()
   end

   ws.onclose = function(req)
      print('/ws/closed')
   end
end)

app.ws('/bench', function(ws)
   ws.onopen = function(req)
      print('/bench/opened')
   end

   ws.onclose = function(req)
      print('/bench/closed')
   end
end)

app.set('public', '.')
app.listen({host="0.0.0.0"})
