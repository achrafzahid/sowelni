/**
 * Strip SSR/server config from angular.json (Angular 19 scaffolds it by default).
 * Run from the Angular project root:  node fix-angular-json.js
 */
const fs = require('fs');
const path = require('path');

const file = path.join(__dirname, 'angular.json');
if (!fs.existsSync(file)) {
  console.error('angular.json not found here.');
  process.exit(1);
}

const config = JSON.parse(fs.readFileSync(file, 'utf8'));
const projectName = Object.keys(config.projects)[0];
const architect =
  config.projects[projectName].architect || config.projects[projectName].targets;

let touched = 0;
if (architect.server) { delete architect.server; touched++; console.log('  ✓ architect.server'); }

const stripSsr = (cfg, label) => {
  if (!cfg) return;
  for (const k of ['server', 'ssr', 'prerender']) {
    if (k in cfg) { delete cfg[k]; touched++; console.log(`  ✓ ${label}.${k}`); }
  }
};
stripSsr(architect.build?.options, 'build.options');
for (const [name, cfg] of Object.entries(architect.build?.configurations || {})) {
  stripSsr(cfg, `build.configurations.${name}`);
}

if (touched === 0) console.log('Already clean.');
else {
  fs.writeFileSync(file, JSON.stringify(config, null, 2) + '\n');
  console.log(`\n✓ angular.json saved (${touched} removed)`);
}
