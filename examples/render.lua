local app = require('../waffle')

app.get('/', function(req, res)
    res.render('main.html')
end)

app.post('/api/v1.0/model', function(req, res)
    local responds = {'aaaa'}
    res.json {models= responds}
end)

app.listen()