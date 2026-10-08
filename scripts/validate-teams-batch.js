// Batch team validator: reads JSON lines {"format": id, "team": paste-text},
// writes JSON lines {"valid": bool, "errors": [..]}. Used by src/propose.py Validator.
const {TeamValidator} = require('./dist/sim/team-validator');
const {Teams} = require('./dist/sim/teams');
const readline = require('readline');

const rl = readline.createInterface({input: process.stdin, terminal: false});
rl.on('line', (line) => {
	let out;
	try {
		const req = JSON.parse(line);
		const validator = TeamValidator.get(req.format);
		const team = Teams.import(req.team);
		const errors = team ? validator.validateTeam(team) : ['could not parse team'];
		out = {valid: !errors, errors: errors || []};
	} catch (e) {
		out = {valid: false, errors: [String(e && e.message || e)]};
	}
	process.stdout.write(JSON.stringify(out) + '\n');
});
