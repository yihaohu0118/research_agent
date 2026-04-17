const CONFIG = {
  portraitsBase: "/static/portraits/",
  portraitCount: 15,
  travelDuration: 260,
};

const STORAGE_KEYS = {
  AGENT_CONFIGS: "AgentConfigs.v1",
  GAME_OPTIONS: "GameOptions.v1",
  WEB_CONFIG_LOADED: "WebConfigLoaded.v1",
  LAST_GAME_OPTIONS: "LastGameOptions.v1",
  CONFIG_UPDATE_TIME: "ConfigUpdateTime.v1",
};

function loadAgentConfigs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.AGENT_CONFIGS);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveAgentConfigs(configs) {
  try {
    localStorage.setItem(STORAGE_KEYS.AGENT_CONFIGS, JSON.stringify(configs));
  } catch (e) {
    console.error("Failed to save agent configs:", e);
  }
}

function loadGameOptions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.GAME_OPTIONS);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveGameOptions(options) {
  try {
    localStorage.setItem(STORAGE_KEYS.GAME_OPTIONS, JSON.stringify(options));
  } catch (e) {
    console.error("Failed to save game options:", e);
  }
}

function shouldLoadWebConfig() {
  const loaded = localStorage.getItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
  return !loaded || loaded !== "true";
}

function markWebConfigLoaded() {
  localStorage.setItem(STORAGE_KEYS.WEB_CONFIG_LOADED, "true");
}

async function loadWebConfig() {
  if (window.location.protocol === "file:") {
    return;
  }
  
  const fromCharacterConfig = sessionStorage.getItem('fromCharacterConfig');
  if (fromCharacterConfig === 'true') {
    sessionStorage.removeItem('fromCharacterConfig');
    
    const hasLoaded = shouldLoadWebConfig() === false;
    if (hasLoaded) {
      const existingConfigs = loadAgentConfigs();
      const configCount = Object.keys(existingConfigs).length;
      if (configCount >= 1) {
        return;
      }
      localStorage.removeItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
      
    }
  } else {
    localStorage.removeItem(STORAGE_KEYS.AGENT_CONFIGS)
    localStorage.removeItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
  }
  
  try {
    const resp = await fetch("/api/options");
    if (!resp.ok) {
      console.warn("Failed to fetch web config:", resp.status);
      return;
    }
    
    const webOpts = await resp.json();
    const portraits = webOpts.portraits || {};
    const defaultModel = webOpts.default_model || {};
    
    const existingConfigs = loadAgentConfigs();
    let updated = false;
    
    for (let id = 1; id <= CONFIG.portraitCount; id++) {
      if (!existingConfigs[id]) {
        existingConfigs[id] = {};
      }
      
      const portraitCfg = portraits[id];
      const hasPortraitCfg = portraitCfg && typeof portraitCfg === "object";
      
      if (hasPortraitCfg && portraitCfg.name) {
        if (!existingConfigs[id].name) {
          existingConfigs[id].name = portraitCfg.name;
          updated = true;
        }
      }
      
      let modelName = null;
      if (hasPortraitCfg && portraitCfg.model && portraitCfg.model.model_name) {
        modelName = portraitCfg.model.model_name;
      } else if (defaultModel.model_name) {
        modelName = defaultModel.model_name;
      }
      if (modelName && !existingConfigs[id].base_model) {
        existingConfigs[id].base_model = modelName;
        updated = true;
      }
      
      let apiBase = null;
      if (hasPortraitCfg && portraitCfg.model) {
        apiBase = portraitCfg.model.url || portraitCfg.model.api_base || null;
      }
      if (!apiBase && defaultModel.api_base) {
        apiBase = defaultModel.api_base;
      }
      if (apiBase && !existingConfigs[id].api_base) {
        existingConfigs[id].api_base = apiBase;
        updated = true;
      }
      
      let apiKey = null;
      if (hasPortraitCfg && portraitCfg.model && portraitCfg.model.api_key) {
        apiKey = portraitCfg.model.api_key;
      } else if (defaultModel.api_key) {
        apiKey = defaultModel.api_key;
      }
      if (apiKey && !existingConfigs[id].api_key) {
        existingConfigs[id].api_key = apiKey;
        updated = true;
      }
      
      let agentClass = null;
      if (hasPortraitCfg && portraitCfg.agent && portraitCfg.agent.type) {
        agentClass = portraitCfg.agent.type;
      } else if (defaultModel.agent_class) {
        agentClass = defaultModel.agent_class;
      }
      if (agentClass && !existingConfigs[id].agent_class) {
        existingConfigs[id].agent_class = agentClass;
        updated = true;
      }
    }
    
    if (updated) {
      saveAgentConfigs(existingConfigs);
    }
    
    markWebConfigLoaded();
  } catch (e) {
    console.warn("Failed to load web config:", e);
  }
}


const AVALON_ROLE_MAP = {
  "Merlin": 0,
  "Percival": 1,
  "Servant": 2,
  "Minion": 3,
  "Assassin": 4,
};
const AVALON_GOOD_ROLES = ["Merlin", "Percival", "Servant"];

const state = {
  selectedIds: new Set(),
  selectedIdsOrder: [],
  selectedGame: "",
  selectedMode: "observe",
  diplomacyOptions: null,
  avalonOptions: null,
  diplomacyPowerOrder: null,
  avalonRoleOrder: null,
  avalonPreviewRoles: null,  // [{role_id, role_name, is_good}, ...]
  diplomacyPreviewPowers: null,  // [power_name, ...]
};

let DOM = {};

function polarPositions(count, radiusX, radiusY) {
  return Array.from({ length: count }).map((_, i) => {
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
  });
}

function computeRedirectUrl(game, mode) {
  if (window.location.protocol !== "file:") return `/${game}/${mode}`;
  return `./static/${game}/${mode}.html`;
}

function addStatusMessage(text) {
  if (!DOM.statusLog) return;

  const bubble = document.createElement("div");
  bubble.className = "status-bubble";
  bubble.textContent = text;
  DOM.statusLog.appendChild(bubble);

  while (DOM.statusLog.children.length > 20) {
    DOM.statusLog.removeChild(DOM.statusLog.firstChild);
  }

  setTimeout(() => {
    if (DOM.statusLog) {
      DOM.statusLog.scrollTop = DOM.statusLog.scrollHeight;
    }
  }, 50);
}

function ensureSeat(id) {
  if (!DOM.tablePlayers) return null;
  
  let seat = DOM.tablePlayers.querySelector(`.seat[data-id="${id}"]`);
  if (seat) return seat;
  
  seat = document.createElement("div");
  seat.className = "seat enter";
  seat.dataset.id = String(id);
  const isHuman = String(id) === "human";
  const src = isHuman ? `${CONFIG.portraitsBase}portrait_human.png` : `${CONFIG.portraitsBase}portrait_${id}.png`;
  const alt = isHuman ? "Human" : `Agent ${id}`;
  
  const AgentConfigs = loadAgentConfigs();
  const cfg = AgentConfigs[id] || {};
  const baseModel = cfg.base_model || "";
  const modelLabel = baseModel ? `<div class="seat-model">${baseModel}</div>` : "";
  
  seat.innerHTML = `
    ${modelLabel}
    <div class="seat-label"></div>
    <img src="${src}" alt="${alt}">
  `;
  seat.style.left = "50%";
  seat.style.top = "50%";
  seat.style.transform = "translate(-50%, -50%) scale(0.8)";
  seat.style.pointerEvents = "auto";
  seat.style.cursor = isHuman ? "default" : "pointer";
  DOM.tablePlayers.appendChild(seat);
  
  requestAnimationFrame(() => seat.classList.remove("enter"));
  return seat;
}

function checkRoleConflict(seatId, newRole, game) {
  if (!game) return null;
  
  const seats = DOM.tablePlayers.querySelectorAll(".seat");
  const currentSelections = [];
  
  seats.forEach(seat => {
    const select = seat.querySelector(".seat-label select");
    if (!select) return;
    let value = select.value;
    if (seat.dataset.id === seatId) {
      value = newRole;
    }
    if (value && value !== "") {
      currentSelections.push(value);
    }
  });
  
  let expectedList = [];
  if (game === "avalon") {
    if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
      expectedList = state.avalonOptions.roles.slice();
    } else {
      expectedList = ["Merlin", "Servant", "Servant", "Minion", "Assassin"];
    }
  } else if (game === "diplomacy") {
    if (state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers)) {
      expectedList = state.diplomacyOptions.powers.slice();
    }
  }
  
  if (expectedList.length === 0) return null;
  
  const counts = {};
  currentSelections.forEach(role => {
    counts[role] = (counts[role] || 0) + 1;
  });
  
  for (const [role, count] of Object.entries(counts)) {
    if (count > 1) {
      return {
        hasConflict: true,
        message: `${role} appears ${count} times! Each role/power should appear exactly once.`,
        conflicts: []
      };
    }
  }
  
  const missing = expectedList.filter(role => !currentSelections.includes(role));
  if (missing.length > 0) {
    return {
      hasConflict: true,
      message: `Missing roles/powers: ${missing.join(", ")}`,
      conflicts: []
    };
  }
  
  return null;
}

function setSeatLabelBySeatId(seatId, text, options = []) {
  const el = DOM.tablePlayers && DOM.tablePlayers.querySelector(`.seat[data-id="${seatId}"]`);
  if (!el) return;
  const labelContainer = el.querySelector(".seat-label");
  if (!labelContainer) return;
  
  if (!text && options.length === 0) {
    el.classList.remove("has-label");
    labelContainer.innerHTML = "";
    return;
  }
  

  if (options.length > 0) {
    const select = document.createElement("select");
    let currentValue = text || options[0];
    options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (opt === currentValue) {
        option.selected = true;
      }
      select.appendChild(option);
    });
    
    select.addEventListener("click", (e) => {
      e.stopPropagation();
    });
    
    select.addEventListener("mousedown", (e) => {
      e.stopPropagation();
    });
    
    select.addEventListener("change", (e) => {
      e.stopPropagation();
      const newRole = e.target.value;
      const game = state.selectedGame;
      
      const conflict = checkRoleConflict(seatId, newRole, game);
      if (conflict && conflict.hasConflict) {
        addStatusMessage(`⚠ ${conflict.message}`);
      }
      
      currentValue = newRole;
      
      if (game === "avalon") {
        const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
        const idx = ids.indexOf(parseInt(seatId, 10));
        if (idx !== -1 && state.avalonRoleOrder) {
          state.avalonRoleOrder[idx] = newRole;
          state.avalonPreviewRoles = state.avalonRoleOrder.map((roleName, i) => ({
            role_id: AVALON_ROLE_MAP[roleName] || 0,
            role_name: roleName,
            is_good: AVALON_GOOD_ROLES.includes(roleName),
          }));
        }
      } else if (game === "diplomacy") {
        const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
        const idx = ids.indexOf(parseInt(seatId, 10));
        if (idx !== -1 && state.diplomacyPowerOrder) {
          state.diplomacyPowerOrder[idx] = newRole;
          state.diplomacyPreviewPowers = state.diplomacyPowerOrder.slice();
        }
      }
      
      updateSelectionHint();
      
      updateTableRoleStats();
      
      addStatusMessage(`Updated role: ${newRole}`);
    });
    
    labelContainer.innerHTML = "";
    labelContainer.appendChild(select);
    el.classList.add("has-label");
  } else {
    labelContainer.textContent = String(text);
  el.classList.add("has-label");
  }
}

function shouldShowPreview() {
  return state.selectedMode !== "participate";
}

function setRandomButtonsEnabled() {
  const disabled = state.selectedMode === "participate";
  const aBtn = document.getElementById("avalon-reroll-roles");
  const dBtn = document.getElementById("diplomacy-shuffle-powers");
  if (aBtn) aBtn.disabled = disabled;
  if (dBtn) dBtn.disabled = disabled;
}

function requiredCountForPreview() {
  const game = state.selectedGame;
  if (!game) return 0;
  if (game === "avalon") return 5;
  if (game === "diplomacy") return 7;
  return 0;
}

function avalonAssignRolesFor5() {
  if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
    const roles = state.avalonOptions.roles.slice();
    return shuffleInPlace(roles);
  }
  
  const roles = ["Merlin", "Servant", "Servant", "Minion", "Assassin"];
  return shuffleInPlace(roles.slice());
}

function updateTableHeadPreview() {
  if (!DOM.tablePlayers) return;

  Array.from(DOM.tablePlayers.querySelectorAll(".seat")).forEach(seat => {
    const label = seat.querySelector(".seat-label");
    if (label) label.innerHTML = "";
    seat.classList.remove("has-label");
  });

  setRandomButtonsEnabled();
  
  const game = state.selectedGame;
  const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
  const isParticipate = state.selectedMode === "participate";
  
  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    const required = isParticipate ? (numPlayers - 1) : numPlayers;
    if (ids.length !== required) {
      updateSelectionHint();
      return;
    }
  } else if (game === "diplomacy") {
    const required = isParticipate ? 6 : 7;
    if (ids.length !== required) {
      updateSelectionHint();
      return;
    }
  }

  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    if (!state.avalonRoleOrder || state.avalonRoleOrder.length !== numPlayers) {
      if (numPlayers === 5) {
      state.avalonRoleOrder = avalonAssignRolesFor5();
      } else {
        state.avalonRoleOrder = Array(numPlayers).fill("Servant");
      }
    }
    state.avalonPreviewRoles = state.avalonRoleOrder.map((roleName, idx) => ({
      role_id: AVALON_ROLE_MAP[roleName] || 0,
      role_name: roleName,
      is_good: AVALON_GOOD_ROLES.includes(roleName),
    }));
    
    if (shouldShowPreview()) {
      let allRoles = [];
      if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
        allRoles = [...new Set(state.avalonOptions.roles)];
      } else {
        allRoles = numPlayers === 5 
          ? ["Merlin", "Servant", "Minion", "Assassin"]
          : ["Servant"];
      }
      ids.forEach((portraitId, idx) => {
        setSeatLabelBySeatId(String(portraitId), state.avalonRoleOrder[idx], allRoles);
      });
    }
  } else if (game === "diplomacy") {
    const powers = (state.diplomacyPowerOrder && state.diplomacyPowerOrder.length === 7)
      ? state.diplomacyPowerOrder
      : (state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers) ? state.diplomacyOptions.powers.slice() : []);
    if (powers.length !== 7) {
      updateSelectionHint();
      return;
    }
    state.diplomacyPreviewPowers = powers.slice();
    
    if (shouldShowPreview()) {
    ids.forEach((portraitId, idx) => {
        const allPowers = state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers) 
          ? state.diplomacyOptions.powers.slice() 
          : powers;
        setSeatLabelBySeatId(String(portraitId), powers[idx], allPowers);
      });
    }
  }
  
  updateSelectionHint();
  
  updateTableRoleStats();
}

function updateTableRoleStats() {
  const statsEl = document.getElementById("table-role-stats");
  if (!statsEl) return;
  
  const game = state.selectedGame;
  if (!game) {
    statsEl.classList.remove("show");
    return;
  }
  
  const mode = state.selectedMode;
  const selected = state.selectedIds.size;
  let required = 0;
  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    required = (mode === "participate") ? (numPlayers - 1) : numPlayers;
  } else if (game === "diplomacy") {
    required = (mode === "participate") ? 6 : 7;
  }
  
  if (selected !== required) {
    statsEl.classList.remove("show");
    return;
  }
  
  let currentSelections = [];
  if (game === "avalon" && state.avalonRoleOrder && Array.isArray(state.avalonRoleOrder)) {
    currentSelections = state.avalonRoleOrder.slice();
  } else if (game === "diplomacy" && state.diplomacyPowerOrder && Array.isArray(state.diplomacyPowerOrder)) {
    currentSelections = state.diplomacyPowerOrder.slice();
  } else {
    const seats = DOM.tablePlayers ? DOM.tablePlayers.querySelectorAll(".seat") : [];
    seats.forEach(seat => {
      const select = seat.querySelector(".seat-label select");
      if (select && select.value) {
        currentSelections.push(select.value);
      }
    });
  }
  
  if (currentSelections.length === 0) {
    statsEl.classList.remove("show");
    return;
  }
  
  const counts = {};
  currentSelections.forEach(role => {
    counts[role] = (counts[role] || 0) + 1;
  });
  
let roleConfig = [];
if (game === "avalon") {
  const numPlayersEl = document.getElementById("avalon-num-players");
  const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
  
  if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
      const roles = state.avalonOptions.roles;
      const roleCounts = {};
      roles.forEach(role => {
        roleCounts[role] = (roleCounts[role] || 0) + 1;
      });
      roleConfig = Object.entries(roleCounts).map(([name, expected]) => ({
        name,
        expected
      }));
    } else {
      if (numPlayers === 5) {
        roleConfig = [
          { name: "Merlin", expected: 1 },
          { name: "Servant", expected: 2 },
          { name: "Minion", expected: 1 },
          { name: "Assassin", expected: 1 }
        ];
      }
    }
  } else if (game === "diplomacy") {
    if (state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers)) {
      roleConfig = state.diplomacyOptions.powers.map(power => ({
        name: power,
        expected: 1
      }));
    }
  }
  
  if (roleConfig.length === 0) {
    statsEl.classList.remove("show");
    return;
  }
  
  statsEl.innerHTML = "";
  roleConfig.forEach(role => {
    const current = counts[role.name] || 0;
    const item = document.createElement("div");
    item.className = "role-stat-item";
    item.innerHTML = `
      <span class="role-name">${role.name}:</span>
      <span class="role-count">${current}/${role.expected}</span>
    `;
    if (current !== role.expected) {
      item.querySelector(".role-count").style.color = "#ff6b6b";
    }
    statsEl.appendChild(item);
  });
  
  statsEl.classList.add("show");
}

function layoutTablePlayers() {
  if (!DOM.tablePlayers) return;
  
  const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
  const wantHuman = state.selectedMode === "participate";
  
  let keys = [];
  if (wantHuman) {
    const game = state.selectedGame;
    let userAgentId = 0;
    
    if (game === "avalon") {
      const userAgentEl = document.getElementById("avalon-user-agent-id");
      userAgentId = userAgentEl ? parseInt(userAgentEl.value, 10) : 0;
    } else if (game === "diplomacy") {
      const hpEl = document.getElementById("diplomacy-human-power");
      const humanPower = hpEl ? hpEl.value : "";
      if (state.diplomacyOptions && state.diplomacyOptions.powers) {
        userAgentId = state.diplomacyOptions.powers.indexOf(humanPower);
        if (userAgentId === -1) userAgentId = 0;
      }
    }
    
    const totalPlayers = ids.length + 1;  // AI + human
    keys = [];
    for (let i = 0; i < totalPlayers; i++) {
      if (i === userAgentId) {
        keys.push("human");
      } else {
        const aiIndex = i < userAgentId ? i : (i - 1);
        if (aiIndex < ids.length) {
          keys.push(String(ids[aiIndex]));
        }
      }
    }
  } else {  
    keys = ids.map((x) => String(x));
  }
  
  const keySet = new Set(keys);
  
  Array.from(DOM.tablePlayers.querySelectorAll(".seat")).forEach(el => {
    const key = String(el.dataset.id || "");
    if (!keySet.has(key)) {
      el.classList.add("leave");
      el.addEventListener("transitionend", () => el.remove(), { once: true });
    }
  });
  
  if (!keys.length) return;
  
  keys.forEach(id => ensureSeat(id));
  
  const rect = DOM.tablePlayers.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const seatSize = 70;
  const radiusX = Math.min(280, Math.max(150, rect.width * 0.35));
  const radiusY = Math.min(125, Math.max(90, rect.height * 0.35));
  const positions = polarPositions(keys.length, radiusX, radiusY);
  
  keys.forEach((id, i) => {
    const el = DOM.tablePlayers.querySelector(`.seat[data-id="${id}"]`);
    if (!el) return;
    
    el.style.left = `${cx + positions[i].x - seatSize / 2}px`;
    el.style.top = `${cy + positions[i].y - seatSize / 2}px`;
    el.style.transform = `rotate(${(i % 2 ? 1 : -1) * 2}deg)`;
    el.style.zIndex = "1";
    el.style.cursor = id === "human" ? "default" : "pointer";
    el.style.pointerEvents = "auto";
    
    if (id !== "human" && !el.dataset.hasEvents) {
      el.dataset.hasEvents = "true";
      
      el.addEventListener("click", (e) => {
        if (e.target.closest(".seat-label") || e.target.closest("select")) {
          return;
        }
        e.stopPropagation();
        e.preventDefault();
        const portraitId = parseInt(id, 10);
        if (!isNaN(portraitId)) {
          const portraitCard = DOM.strip?.querySelector(`.portrait-card[data-id="${portraitId}"]`);
          let agentName = `Agent ${portraitId}`;
          if (portraitCard) {
            const nameEl = portraitCard.querySelector(".portrait-name");
            if (nameEl) {
              agentName = nameEl.textContent.trim();
            }
          }
          toggleAgent({ id: portraitId, name: agentName }, null);
        }
      });
    }
  });

  updateTableHeadPreview();
}

function updateCounter() {
  if (DOM.counterEl) {
    const game = state.selectedGame;
    const mode = state.selectedMode;
    const selected = state.selectedIds.size;
    let required = 0;
    
    if (game === 'avalon') {
      const numPlayersEl = document.getElementById("avalon-num-players");
      const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
      required = (mode === 'participate') ? (numPlayers - 1) : numPlayers;
    } else if (game === 'diplomacy') {
      required = (mode === 'participate') ? 6 : 7;
    }
    
    DOM.counterEl.textContent = `${selected}/${required}`;
  }
  updateSelectionHint();
  updateTableRoleStats();
}

function checkConfigError() {
  const game = state.selectedGame;
  const mode = state.selectedMode;
  const selected = state.selectedIds.size;
  
  let required = 0;
  if (game === 'avalon') {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    required = (mode === 'participate') ? (numPlayers - 1) : numPlayers;
  } else if (game === 'diplomacy') {
    required = (mode === 'participate') ? 6 : 7;
  }
  
  if (selected !== required) {
    return {
      hasError: true,
      message: selected < required ? `${required - selected} more` : `⚠ Exceed ${selected - required}`
    };
  }
  
  const currentMode = state.selectedMode;
  const isParticipate = currentMode === "participate";
  
  let currentSelections = [];
  
  if (isParticipate) {
    if (game === "avalon" && state.avalonRoleOrder && Array.isArray(state.avalonRoleOrder)) {
      currentSelections = state.avalonRoleOrder.slice();
    } else if (game === "diplomacy" && state.diplomacyPowerOrder && Array.isArray(state.diplomacyPowerOrder)) {
      currentSelections = state.diplomacyPowerOrder.slice();
    }
  } else {
    if (!DOM.tablePlayers) return null;
    
    const seats = DOM.tablePlayers.querySelectorAll(".seat");
    seats.forEach(seat => {
      const select = seat.querySelector(".seat-label select");
      if (!select) return;
      const value = select.value;
      if (value && value !== "") {
        currentSelections.push(value);
      }
    });
  }
  
  let expectedList = [];
  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    
    if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
      expectedList = state.avalonOptions.roles.slice();
    } else {
      expectedList = ["Merlin", "Servant", "Servant", "Minion", "Assassin"];
    }
  } else if (game === "diplomacy") {
    if (state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers)) {
      expectedList = state.diplomacyOptions.powers.slice();
    }
  }
  
  if (expectedList.length === 0) return null;
  
  if (currentSelections.length === 0) return null;
  
  const currentCounts = {};
  currentSelections.forEach(role => {
    currentCounts[role] = (currentCounts[role] || 0) + 1;
  });
  
  const expectedCounts = {};
  expectedList.forEach(role => {
    expectedCounts[role] = (expectedCounts[role] || 0) + 1;
  });
  
  if (game === "avalon") {
    for (const role of Object.keys(currentCounts)) {
      if (!(role in expectedCounts)) {
        return {
          hasError: true,
          message: `Config Error`
        };
      }
    }
    
    for (const role of Object.keys(expectedCounts)) {
      const currentCount = currentCounts[role] || 0;
      const expectedCount = expectedCounts[role];
      if (currentCount !== expectedCount) {
        return {
          hasError: true,
          message: `Config Error`
        };
      }
    }
  } else if (game === "diplomacy") {
    for (const [role, count] of Object.entries(currentCounts)) {
      if (count > 1) {
        return {
          hasError: true,
          message: `Config Error`
        };
      }
    }
    
    const missing = expectedList.filter(role => !currentSelections.includes(role));
    if (missing.length > 0) {
      return {
        hasError: true,
        message: `Config Error`
      };
    }
    
    const extra = currentSelections.filter(role => !expectedList.includes(role));
    if (extra.length > 0) {
      return {
        hasError: true,
        message: `Config Error`
      };
    }
  }
  
  return null;
}

function updateSelectionHint() {
  const game = state.selectedGame;
  const mode = state.selectedMode;
  const selected = state.selectedIds.size;
  const hintPill = document.getElementById('selection-hint-pill');
  const hintEl = document.getElementById('selection-hint');
  
  if (!hintPill || !hintEl) return;
  
  let hint = '';
  let showHint = false;
  let required = 0;
  
  if (game === 'avalon') {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    required = (mode === 'participate') ? (numPlayers - 1) : numPlayers;
    showHint = true;
  } else if (game === 'diplomacy') {
    required = (mode === 'participate') ? 6 : 7;
    showHint = true;
  }
  
  if (showHint) {
    const configError = checkConfigError();
    
    if (configError && configError.hasError) {
      hint = configError.message;
      hintPill.style.borderColor = '#ff6b6b';
    } else if (selected < required) {
      hint = `${required - selected} more`;
      hintPill.style.borderColor = '#ffdd57';
    } else if (selected === required) {
      hint = `✓ Correct`;
      hintPill.style.borderColor = '#51f6a5';
    } else {
      hint = `⚠ Exceed ${selected - required}`;
      hintPill.style.borderColor = '#ff6b6b';
  }
  
    hintEl.textContent = hint;
    hintPill.style.display = 'inline-flex';
  } else {
    hintPill.style.display = 'none';
  }
}

function updatePortraitCardActiveState(portraitId, isActive) {
  if (!DOM.strip) return;
  const card = DOM.strip.querySelector(`.portrait-card[data-id="${portraitId}"]`);
  if (card) {
    if (isActive) {
      card.classList.add("active");
    } else {
      card.classList.remove("active");
    }
  }
}

function toggleAgent(person, card) {
  const existed = state.selectedIds.has(person.id);
  
  if (existed) {
    state.selectedIds.delete(person.id);
    const idx = state.selectedIdsOrder.indexOf(person.id);
    if (idx !== -1) {
      state.selectedIdsOrder.splice(idx, 1);
    }
    if (card) {
      card.classList.remove("active");
    } else {
      updatePortraitCardActiveState(person.id, false);
    }
    updateCounter();
    layoutTablePlayers();
    addStatusMessage(`${person.name} left the team!`); 
    return;
  }
  
  state.selectedIds.add(person.id);
  if (!state.selectedIdsOrder.includes(person.id)) {
    state.selectedIdsOrder.push(person.id);
  }
  if (card) {
    card.classList.add("active");
  } else {
    updatePortraitCardActiveState(person.id, true);
  }
  updateCounter();
  layoutTablePlayers();
  addStatusMessage(`${person.name} joined the team!`);
}

function renderPortraits() {
  if (!DOM.portraitsGrid) return;
  
  DOM.portraitsGrid.innerHTML = "";
  
  const AgentConfigs = loadAgentConfigs();
  
  const portraits = Array.from({ length: CONFIG.portraitCount }).map((_, i) => {
    const id = i + 1;
    const cfg = AgentConfigs[id] || {};
    return {
      id,
      name: cfg.name || `Agent ${id}`,
      src: `${CONFIG.portraitsBase}portrait_${id}.png`,
      base_model: cfg.base_model || "",
    };
  });
  
  portraits.forEach(p => {
    const card = document.createElement("div");
    card.className = "portrait-card";
    if (state.selectedIds.has(p.id)) {
      card.classList.add("active");
    }
    card.dataset.id = String(p.id);
    
    const modelLabel = p.base_model 
      ? `<div class="portrait-model">${p.base_model}</div>` 
      : "";
    
    card.innerHTML = `
      ${modelLabel}
      <img src="${p.src}" alt="${p.name}">
      <div class="portrait-name">${p.name}</div>
    `;
    card.addEventListener("click", () => toggleAgent(p, card));
    DOM.portraitsGrid.appendChild(card);
  });
}

function focusGame(game) {
  if (!DOM.gameCards) return;
  DOM.gameCards.forEach(c => c.classList.toggle("active", c.dataset.game === game));
}

function setGame(game) {
  state.selectedGame = game || "";
  focusGame(state.selectedGame);
  
  if (!state.selectedGame) {
    if (DOM.avalonFields) DOM.avalonFields.classList.remove("show");
    if (DOM.diplomacyFields) DOM.diplomacyFields.classList.remove("show");
    updateCounter();
    updateTableRoleStats();
    const tablePreview = document.getElementById("table-preview");
    if (tablePreview) {
      tablePreview.classList.remove("has-game");
    }
    return;
  }
  
  addStatusMessage(`Selected game: ${state.selectedGame}`);
  
  if (state.selectedGame === "diplomacy") {
    const lastGame = localStorage.getItem(STORAGE_KEYS.LAST_GAME_OPTIONS);
    const forceRefresh = lastGame !== "diplomacy";
    if (forceRefresh) {
      localStorage.setItem(STORAGE_KEYS.LAST_GAME_OPTIONS, "diplomacy");
    }
    fetchDiplomacyOptions(forceRefresh).then(() => {
      updateCounter();
    });
  } else if (state.selectedGame === "avalon") {
    const lastGame = localStorage.getItem(STORAGE_KEYS.LAST_GAME_OPTIONS);
    const forceRefresh = lastGame !== "avalon";
    if (forceRefresh) {
      localStorage.setItem(STORAGE_KEYS.LAST_GAME_OPTIONS, "avalon");
      fetchAvalonOptions(forceRefresh).then(() => {
        updateCounter();
      });
    }
    updateCounter();
  }
  
  updateConfigVisibility();
  updateSelectionHint();
  updateTableHeadPreview();
  updateTableRoleStats();
  
  const tablePreview = document.getElementById("table-preview");
  if (tablePreview) {
    if (state.selectedGame) {
      tablePreview.classList.add("has-game");
      setTimeout(() => {
        layoutTablePlayers();
      }, 0);
    } else {
      tablePreview.classList.remove("has-game");
    }
  }
}

function setMode(mode) {
  state.selectedMode = mode || "observe";
  
  if (DOM.modeLabelEl) {
    DOM.modeLabelEl.textContent = state.selectedMode === "observe" ? "Observer" : "Participate";
  }
  
  if (DOM.modeToggle) {
    DOM.modeToggle.querySelectorAll(".mode-opt").forEach(opt => {
      opt.classList.toggle("active", opt.dataset.mode === state.selectedMode);
    });
  }
  
  updateConfigVisibility();
  updateSelectionHint();
  layoutTablePlayers();
  updateTableHeadPreview();
  updateTableRoleStats();
}

function updateConfigVisibility() {
  const game = state.selectedGame;
  const mode = state.selectedMode;
  
  if (DOM.avalonFields) {
    DOM.avalonFields.classList.toggle("show", game === "avalon" && !!mode);
  }
  if (DOM.diplomacyFields) {
    DOM.diplomacyFields.classList.toggle("show", game === "diplomacy" && !!mode);
  }
  
  document.querySelectorAll(".avalon-participate-only").forEach(el => {
    el.style.display = (game === "avalon" && mode === "participate") ? "flex" : "none";
  });
  
  document.querySelectorAll(".diplomacy-participate-only").forEach(el => {
    el.style.display = (game === "diplomacy" && mode === "participate") ? "flex" : "none";
  });
  
  if (DOM.powerModelsSection) {
    DOM.powerModelsSection.style.display = (game === "diplomacy" && state.diplomacyOptions) ? "block" : "none";
  }
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function fetchAvalonOptions(forceRefresh = false) {
  try {
    if (window.location.protocol === "file:") {
      state.avalonOptions = null;
      updateConfigVisibility();
      return;
    }
    
    const cachedOptions = loadGameOptions();
    if (!forceRefresh && cachedOptions.avalon) {
      state.avalonOptions = cachedOptions.avalon;
    } else {
      const resp = await fetch("/api/options?game=avalon");
      if (!resp.ok) throw new Error("Failed to fetch options");
      state.avalonOptions = await resp.json();
      
      const gameOptions = loadGameOptions();
      gameOptions.avalon = state.avalonOptions;
      saveGameOptions(gameOptions);
    }
    
    if (state.avalonOptions.defaults) {
      const numPlayersEl = document.getElementById("avalon-num-players");
      const langEl = document.getElementById("avalon-language");
      
      if (numPlayersEl) numPlayersEl.value = state.avalonOptions.defaults.num_players;
      if (langEl) langEl.value = state.avalonOptions.defaults.language;
    }
    
    updateConfigVisibility();
    updateTableHeadPreview();
  } catch (e) {
    console.error("Failed to fetch avalon options:", e);
    state.avalonOptions = null;
    updateConfigVisibility();
  }
}

async function fetchDiplomacyOptions(forceRefresh = false) {
  try {
    if (window.location.protocol === "file:") {
      state.diplomacyOptions = null;
      updateConfigVisibility();
      return;
    }
    
    const cachedOptions = loadGameOptions();
    if (!forceRefresh && cachedOptions.diplomacy) {
      state.diplomacyOptions = cachedOptions.diplomacy;
      state.diplomacyPowerOrder = Array.isArray(state.diplomacyOptions.powers) ? state.diplomacyOptions.powers.slice() : null;
    } else {
      const resp = await fetch("/api/options?game=diplomacy");
      if (!resp.ok) throw new Error("Failed to fetch options");
      state.diplomacyOptions = await resp.json();
      state.diplomacyPowerOrder = Array.isArray(state.diplomacyOptions.powers) ? state.diplomacyOptions.powers.slice() : null;
      
      const gameOptions = loadGameOptions();
      gameOptions.diplomacy = state.diplomacyOptions;
      saveGameOptions(gameOptions);
    }
    
    const hpSelect = document.getElementById("diplomacy-human-power");
    if (hpSelect && state.diplomacyOptions.powers) {
      hpSelect.innerHTML = "";
      state.diplomacyOptions.powers.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        hpSelect.appendChild(opt);
      });
    }
    
    if (state.diplomacyOptions.defaults) {
      const maxPhasesEl = document.getElementById("diplomacy-max-phases");
      const negRoundsEl = document.getElementById("diplomacy-negotiation-rounds");
      const langEl = document.getElementById("diplomacy-language");
      
      if (maxPhasesEl) maxPhasesEl.value = state.diplomacyOptions.defaults.max_phases;
      if (negRoundsEl) negRoundsEl.value = state.diplomacyOptions.defaults.negotiation_rounds;
      if (langEl) langEl.value = state.diplomacyOptions.defaults.language;
    }
    
    updateConfigVisibility();
    updateTableHeadPreview();
  } catch (e) {
    console.error("Failed to fetch diplomacy options:", e);
    state.diplomacyOptions = null;
    updateConfigVisibility();
  }
}

function buildPayload(game, mode) {
  const payload = { game, mode };
  
  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const languageEl = document.getElementById("avalon-language");
    const userAgentEl = document.getElementById("avalon-user-agent-id");
    
    payload.num_players = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    payload.language = languageEl ? languageEl.value : "en";
    if (mode === "participate" && userAgentEl) {
      payload.user_agent_id = parseInt(userAgentEl.value, 10);
    }
    
    if (state.avalonPreviewRoles && state.avalonPreviewRoles.length > 0) {
      payload.preset_roles = state.avalonPreviewRoles;
    }
    
    payload.selected_portrait_ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
    
    const agentConfigs = loadAgentConfigs();
    const agent_configs = {};
    payload.selected_portrait_ids.forEach(portraitId => {
      const config = agentConfigs[portraitId];
      if (config && (config.base_model || config.api_base || config.api_key || config.agent_class)) {
        agent_configs[portraitId] = {
          base_model: config.base_model || "",
          api_base: config.api_base || "",
          api_key: config.api_key || "",
          agent_class: config.agent_class || "",
        };
      }
    });
    if (Object.keys(agent_configs).length > 0) {
      payload.agent_configs = agent_configs;
    }
    
  } else if (game === "diplomacy") {
    const maxPhasesEl = document.getElementById("diplomacy-max-phases");
    const negRoundsEl = document.getElementById("diplomacy-negotiation-rounds");
    const langEl = document.getElementById("diplomacy-language");
    const hpEl = document.getElementById("diplomacy-human-power");
    
    payload.max_phases = maxPhasesEl ? parseInt(maxPhasesEl.value, 10) : 20;
    payload.negotiation_rounds = negRoundsEl ? parseInt(negRoundsEl.value, 10) : 3;
    payload.language = langEl ? langEl.value : "en";
    
    if (mode === "participate" && hpEl && hpEl.value) {
      payload.human_power = hpEl.value;
    }
    
    if (state.diplomacyPreviewPowers && state.diplomacyPreviewPowers.length > 0) {
      payload.power_names = state.diplomacyPreviewPowers;
    } else if (state.diplomacyOptions && state.diplomacyOptions.powers && state.diplomacyOptions.powers.length === 7) {
      payload.power_names = state.diplomacyOptions.powers.slice();
    }
    
    const agentConfigs = loadAgentConfigs();
    const agent_configs = {};
    const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
    
    let fullPortraitIds = [];
    if (payload.power_names && payload.power_names.length > 0) {
        if (mode === "participate" && payload.human_power) {
          const humanPowerIndex = payload.power_names.indexOf(payload.human_power);
          let aiIndex = 0;
          for (let i = 0; i < payload.power_names.length; i++) {
            if (i === humanPowerIndex) {
              fullPortraitIds.push(-1);
            } else {
              if (aiIndex < ids.length) {
                fullPortraitIds.push(ids[aiIndex]);
                aiIndex++;
              } else {
                fullPortraitIds.push(-1);
              }
            }
          }
        } else {
          fullPortraitIds = ids.slice(0, payload.power_names.length);
          while (fullPortraitIds.length < payload.power_names.length) {
            fullPortraitIds.push(-1);
          }
        }
      } else {
        fullPortraitIds = ids;
      }
      
      if (payload.power_names && fullPortraitIds.length === payload.power_names.length) {
        payload.power_names.forEach((power, index) => {
          const portraitId = fullPortraitIds[index];
          if (portraitId !== -1 && portraitId !== null && portraitId !== undefined) {
            const config = agentConfigs[portraitId];
            if (config && (config.base_model || config.api_base || config.api_key || config.agent_class)) {
              agent_configs[portraitId] = {
                base_model: config.base_model || "",
                api_base: config.api_base || "",
                api_key: config.api_key || "",
                agent_class: config.agent_class || "",
              };
            }
          }
        });
      }
      
      if (Object.keys(agent_configs).length > 0) {
        payload.agent_configs = agent_configs;
      }
      
      payload.selected_portrait_ids = fullPortraitIds;
  }
  
  return payload;
}

function closeModeDropdown() {
  if (DOM.modeToggle) {
    DOM.modeToggle.classList.remove("open");
  }
}

function initEventListeners() {
  if (DOM.gameCards) {
    DOM.gameCards.forEach(card => {
      card.addEventListener("click", () => {
        setGame(card.dataset.game);
      });
    });
  }
  
  if (DOM.modeToggle) {
    const pill = DOM.modeToggle.querySelector(".pill-mode");
    if (pill) {
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        DOM.modeToggle.classList.toggle("open");
      });
    }
    
    DOM.modeToggle.querySelectorAll(".mode-opt").forEach(opt => {
      opt.addEventListener("click", (e) => {
        e.stopPropagation();
        setMode(opt.dataset.mode);
        closeModeDropdown();
        addStatusMessage(`Switched to ${opt.dataset.mode === "observe" ? "observe" : "participate"} mode`);
      });
    });
  }
  
  document.addEventListener("click", () => closeModeDropdown());
  
  const avalonNumPlayers = document.getElementById("avalon-num-players");
  if (avalonNumPlayers) {
    avalonNumPlayers.addEventListener("change", function() {
      const numPlayers = parseInt(this.value, 10);
      const userAgentSelect = document.getElementById("avalon-user-agent-id");
      if (userAgentSelect) {
        userAgentSelect.innerHTML = "";
        for (let i = 0; i < numPlayers; i++) {
          const opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = String(i);
          userAgentSelect.appendChild(opt);
        }
      }
      state.avalonRoleOrder = null;
      updateCounter();
      updateTableHeadPreview();
    });
  }
  
  const avalonUserAgentId = document.getElementById("avalon-user-agent-id");
  if (avalonUserAgentId) {
    avalonUserAgentId.addEventListener("change", function() {
      layoutTablePlayers();
      updateTableHeadPreview();
    });
  }
  
  if (DOM.avalonRerollRolesBtn) {
    DOM.avalonRerollRolesBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (state.selectedMode === "participate") return;
      state.avalonRoleOrder = avalonAssignRolesFor5();
      updateTableHeadPreview();
    });
  }

  const diplomacyHumanPower = document.getElementById("diplomacy-human-power");
  if (diplomacyHumanPower) {
    diplomacyHumanPower.addEventListener("change", function() {
      layoutTablePlayers();
      updateTableHeadPreview();
    });
  }

  if (DOM.diplomacyShufflePowersBtn) {
    DOM.diplomacyShufflePowersBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (state.selectedMode === "participate") return;
      if (!state.diplomacyPowerOrder && state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers)) {
        state.diplomacyPowerOrder = state.diplomacyOptions.powers.slice();
      }
      if (state.diplomacyPowerOrder) shuffleInPlace(state.diplomacyPowerOrder);
      updateTableHeadPreview();
    });
  }
  
  if (DOM.randomSelectBtn) {
    DOM.randomSelectBtn.addEventListener("click", (e) => {
      e.preventDefault();
      
      const game = state.selectedGame;
      const mode = state.selectedMode;
      const required =
        game === "avalon" ? (mode === "participate" ? 4 : 5) :
        game === "diplomacy" ? (mode === "participate" ? 6 : 7) :
        0;
      
      if (!game) {
        addStatusMessage("Please select a game");
        return;
      }
      
      if (required === 0) {
        addStatusMessage("Cannot determine the number of agents to select");
        return;
      }
      
      state.selectedIds.clear();
      state.selectedIdsOrder = [];
      
      if (game === "avalon") {
        state.avalonRoleOrder = null;
      } else if (game === "diplomacy") {
        state.diplomacyPowerOrder = null;
      }
      
      const allIds = Array.from({ length: CONFIG.portraitCount }, (_, i) => i + 1);
      
      const shuffled = allIds.slice();
      shuffleInPlace(shuffled);
      const selected = shuffled.slice(0, required);
      
      selected.forEach(id => {
        state.selectedIds.add(id);
        state.selectedIdsOrder.push(id);
      });
      
      renderPortraits();
      
      updateCounter();
      layoutTablePlayers();
      updateTableHeadPreview();
      
      addStatusMessage(`Randomly selected ${required} agents`);
    });
  }
  
  if (DOM.startBtn) {
    DOM.startBtn.addEventListener("click", async () => {
      const game = state.selectedGame;
      const mode = state.selectedMode;
      const required =
        game === "avalon" ? (mode === "participate" ? 4 : 5) :
        game === "diplomacy" ? (mode === "participate" ? 6 : 7) :
        1;
      
      if (!game) {
        addStatusMessage("Please select a game on the right");
        return;
      }
      if (!mode) {
        addStatusMessage("Please select a mode");
        return;
      }
      if (state.selectedIds.size !== required) {
        addStatusMessage(`Currently selected ${state.selectedIds.size} agents, need to select ${Math.max(0, required - state.selectedIds.size)} more agents`);
        return;
      }
      
      try {
        DOM.startBtn.disabled = true;
        addStatusMessage("Preparing to start...");
        
        const payload = buildPayload(game, mode);
        
        const keysToKeep = ['gameConfig', 'selectedPortraits', 'gameLanguage'];
        Object.keys(sessionStorage).forEach(key => {
          if (!keysToKeep.includes(key)) {
            sessionStorage.removeItem(key);
          }
        });
        
        const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
        let selectedPortraitsArray = [];
        
        if (mode === "participate") {
          let userAgentId = 0;
          let numPlayers = 0;
          
          if (game === "avalon") {
            userAgentId = payload.user_agent_id || 0;
            numPlayers = payload.num_players || 5;
          } else if (game === "diplomacy") {
            const humanPower = payload.human_power || "";
            numPlayers = 7;
            if (state.diplomacyOptions && state.diplomacyOptions.powers) {
              userAgentId = state.diplomacyOptions.powers.indexOf(humanPower);
              if (userAgentId === -1) userAgentId = 0;
            }
          }
          
          for (let i = 0; i < numPlayers; i++) {
            if (i !== userAgentId) {
              const aiIndex = i < userAgentId ? i : (i - 1);
              if (aiIndex < ids.length) {
                selectedPortraitsArray.push(ids[aiIndex]);
              }
            }
          }
        } else {
          selectedPortraitsArray = ids;
        }
        
        sessionStorage.setItem("gameConfig", JSON.stringify(payload));
        sessionStorage.setItem("selectedPortraits", JSON.stringify(selectedPortraitsArray));
        sessionStorage.setItem("gameLanguage", payload.language || "en");
        
        setTimeout(() => {
          const url = computeRedirectUrl(game, mode);
          const timestamp = Date.now();
          const separator = url.includes('?') ? '&' : '?';
          window.location.href = `${url}${separator}_t=${timestamp}`;
        }, CONFIG.travelDuration + 300);
      } catch (e) {
        alert("Failed to start: " + e.message);
        DOM.startBtn.disabled = false;
      }
    });
  }
  
  window.addEventListener("resize", () => layoutTablePlayers());
}

function initDOM() {
  DOM = {
    strip: document.getElementById("portraits-strip"),
    portraitsGrid: document.getElementById("portraits-grid"),
    tablePlayers: document.getElementById("table-players"),
    statusLog: document.getElementById("status-log"),
    counterEl: document.getElementById("counter"),
    modeLabelEl: document.getElementById("mode-label"),
    avalonFields: document.getElementById("avalon-fields"),
    diplomacyFields: document.getElementById("diplomacy-fields"),
    startBtn: document.getElementById("start-btn"),
    powerModelsSection: document.getElementById("power-models-section"),
    powerModelsGrid: document.getElementById("power-models-grid"),
    gameCards: Array.from(document.querySelectorAll(".game-card")),
    modeToggle: document.querySelector(".mode-toggle"),
    selectionHintPill: document.getElementById("selection-hint-pill"),
    selectionHint: document.getElementById("selection-hint"),
    avalonRerollRolesBtn: document.getElementById("avalon-reroll-roles"),
    diplomacyShufflePowersBtn: document.getElementById("diplomacy-shuffle-powers"),
    randomSelectBtn: document.getElementById("random-select-btn"),
  };
}

let lastConfigUpdateTime = localStorage.getItem(STORAGE_KEYS.CONFIG_UPDATE_TIME) || "0";

async function init() {
  initDOM();
  
  await loadWebConfig();
  
  renderPortraits();
  updateCounter();
  layoutTablePlayers();
  
  setMode("observe");
  updateConfigVisibility();
  updateTableHeadPreview();
  
  initEventListeners();
  
  let lastFocusTime = Date.now();
  
  window.addEventListener("focus", () => {
    const now = Date.now();
    if (now - lastFocusTime < 500) return;
    lastFocusTime = now;
    
    const currentUpdateTime = localStorage.getItem(STORAGE_KEYS.CONFIG_UPDATE_TIME) || "0";
    if (currentUpdateTime !== lastConfigUpdateTime) {
      lastConfigUpdateTime = currentUpdateTime;
      renderPortraits();
    }
  });
  
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEYS.AGENT_CONFIGS) {
      renderPortraits();
    }
  });
  
  window.addEventListener('localStorageChange', () => {
    renderPortraits();
  });
  
  addStatusMessage("Welcome to Agent Arena!");
  addStatusMessage("Please select Agents and start the game...");
  
  initStepHints();
  
  blinkStepHints();
}

function initStepHints() {
  const stepHints = document.querySelectorAll(".step-hint");
  stepHints.forEach(hint => {
    const target = hint.dataset.target;
    if (!target) return;
    
    hint.addEventListener("mouseenter", () => {
      let targetEl = null;
      if (target === "games") {
        targetEl = document.getElementById("games");
      } else if (target === "agents") {
        targetEl = document.getElementById("portraits-strip");
      } else if (target === "scene") {
        targetEl = document.getElementById("scene");
      } else if (target === "start-btn") {
        targetEl = document.getElementById("start-btn");
      }
      
      if (targetEl) {
        targetEl.classList.add("highlight");
      }
    });
    
    hint.addEventListener("mouseleave", () => {
      let targetEl = null;
      if (target === "games") {
        targetEl = document.getElementById("games");
      } else if (target === "agents") {
        targetEl = document.getElementById("portraits-strip");
      } else if (target === "scene") {
        targetEl = document.getElementById("scene");
      } else if (target === "start-btn") {
        targetEl = document.getElementById("start-btn");
      }
      
      if (targetEl) {
        targetEl.classList.remove("highlight");
      }
    });
  });
}

function blinkStepHints() {
  const stepHints = document.querySelectorAll(".step-hint");
  stepHints.forEach(hint => {
    hint.classList.add("initial-blink");
    
    hint.addEventListener("animationend", () => {
      hint.classList.remove("initial-blink");
    }, { once: true });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
